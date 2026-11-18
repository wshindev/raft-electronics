#!/usr/bin/env python3
"""Step 07: rerank retrieval candidates with a reranker server.

Reads a candidates file written by `retrieval_eval.py --bm25 --dump-candidates N` (hybrid
top N per question) and scores every (question, candidate chunk) pair with the server's
/v1/rerank endpoint (llama-server started with --reranking, e.g. Qwen3-Reranker GGUF).
Chunks are sent as raw clean text. The server's own rerank template supplies the
instruction: llama.cpp bakes it into the GGUF at conversion (tokenizer.chat_template.rerank,
set in conversion/qwen.py), and /v1/rerank can't override it per request. Note the
instruction in --model-tag. retrieval_eval.py --rerank turns the
output into recall rows.

Usage:
    # sanity check on 40 questions: does reranking move the oracle up? are scores sane?
    uv run python3 -I pipeline/rerank.py check --data data/model --url http://HOST:8080 \\
        --cands data/model/rerank/candidates_raft_heldout_fable.jsonl
    # score everything; questions already in the output are skipped (resumable)
    uv run python3 -I pipeline/rerank.py run --data data/model --url http://HOST:8080 \\
        --cands data/model/rerank/candidates_raft_heldout_fable.jsonl --model-tag qwen3-rerank-4b-bf16

The server needs a context per slot (-c divided by --parallel) and -ub of at least 4,096
tokens: the longest chunk is ~3k tokens plus the question.
If the server needs a key, set LLAMA_API_KEY in the environment (sent as a Bearer token).

Writes to <data>/rerank/<model-tag>/:
    <gen>.jsonl   {qid, base, candidates, scores} per question, candidates in hybrid order
    meta.json     model tag, server model, candidates file, depth
"""
import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("LLAMA_API_KEY", "")
SEED = 7


def headers():
    h = {'Content-Type': 'application/json'}
    if API_KEY:
        h['Authorization'] = f'Bearer {API_KEY}'
    return h


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def post_json(url, path, payload, timeout):
    req = urllib.request.Request(url.rstrip('/') + path, data=json.dumps(payload).encode(),
                                 headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'server error {e.code}: {e.read()[:500]!r}') from None


def server_model(url, timeout):
    try:
        req = urllib.request.Request(url.rstrip('/') + '/v1/models', headers=headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [m.get('id') for m in json.load(r).get('data', [])]
    except (urllib.error.URLError, ValueError):
        return None


def score(args, chunks, rec):
    """Relevance score per candidate, in candidate order."""
    docs = [chunks[c] for c in rec['candidates']]
    res = post_json(args.url, '/v1/rerank', {'model': 'reranker', 'query': rec['question'],
                                             'documents': docs, 'top_n': len(docs)}, args.timeout)['results']
    out = [None] * len(docs)
    for r in res:
        out[r['index']] = r['relevance_score']
    if any(s is None for s in out):
        raise RuntimeError(f'{rec["qid"]}: server returned {len(res)} of {len(docs)} scores')
    return out


def run_all(args, chunks, recs):
    """Yield (rec, scores) as requests finish, args.workers at a time."""
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(score, args, chunks, r): r for r in recs}
        for f in as_completed(futs):
            try:
                yield futs[f], f.result()
            except RuntimeError as e:
                sys.exit(f'{futs[f]["qid"]}: {e}\n'
                         'If the input is too long, raise the server context per slot / -ub.')


def cmd_check(args, chunks, recs):
    print('server models:', server_model(args.url, args.timeout))
    pick = random.Random(SEED).sample(recs, min(args.n, len(recs)))
    hyb, rr = [], []  # 0-based oracle rank among the candidates: hybrid order, reranked order
    lo, hi, flat = float('inf'), float('-inf'), 0
    for rec, s in run_all(args, chunks, pick):
        lo, hi = min(lo, *s), max(hi, *s)
        flat += max(s) - min(s) < 1e-6
        if rec['oracle'] not in rec['candidates']:
            continue
        o = rec['candidates'].index(rec['oracle'])
        hyb.append(o)
        rr.append(sum(x > s[o] for x in s))
    n, depth = len(hyb), len(pick[0]['candidates'])
    print(f'{len(pick)} questions, oracle among the {depth} candidates for {n}')
    for k in (1, 5):
        print(f'oracle in top {k}: hybrid {sum(r < k for r in hyb)}/{n}, reranked {sum(r < k for r in rr)}/{n}')
    print(f'score range {lo:.4g} .. {hi:.4g}; questions with all-equal scores: {flat}')
    print('Oracle-only counts miss near-duplicates that answer equally well; judge with the full run '
          '(retrieval_eval.py --rerank reports quote-equivalent hits).')
    if flat or sum(r < 5 for r in rr) < 2 * n * 5 / depth:  # no better than twice chance at top 5
        print('Scores look wrong (flat, or near chance). Check that the GGUF has the reranker '
              'head/template and that llama-server runs with --reranking.')


def cmd_run(args, chunks, recs):
    gen = os.path.basename(args.cands).removeprefix('candidates_').removesuffix('.jsonl')
    out_dir = os.path.join(args.data, 'rerank', args.model_tag)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{gen}.jsonl')
    done = {r['qid'] for r in load(out)} if os.path.exists(out) else set()
    todo = [r for r in recs if r['qid'] not in done]
    print(f'{len(done)} questions already scored, {len(todo)} to go')
    with open(out, 'a') as f:
        for n, (rec, s) in enumerate(run_all(args, chunks, todo), 1):
            f.write(json.dumps({'qid': rec['qid'], 'base': rec['base'], 'candidates': rec['candidates'],
                                'scores': s}) + '\n')
            f.flush()
            print(f'\r  {n}/{len(todo)}', end='', flush=True)
    print()
    meta = {'model_tag': args.model_tag, 'server_models': server_model(args.url, args.timeout),
            'candidates_file': args.cands, 'depth': len(recs[0]['candidates'])}
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='data/<source> folder')
        p.add_argument('--url', required=True, help='reranker server base URL, e.g. http://localhost:8080')
        p.add_argument('--cands', required=True, help='candidates file from retrieval_eval.py --dump-candidates')
        p.add_argument('--workers', type=int, default=4, help='concurrent requests (match the server --parallel)')
        p.add_argument('--timeout', type=float, default=600)
        if name == 'check':
            p.add_argument('--n', type=int, default=40, help='sample questions')
        else:
            p.add_argument('--model-tag', required=True, help='names the output folder, e.g. qwen3-rerank-4b-bf16')
    args = ap.parse_args()
    chunks = {c['id']: c['text'] for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    recs = load(args.cands)
    {'check': cmd_check, 'run': cmd_run}[args.cmd](args, chunks, recs)


if __name__ == '__main__':
    main()
