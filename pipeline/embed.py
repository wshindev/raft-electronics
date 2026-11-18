#!/usr/bin/env python3
"""Step 6b-1: embed chunks, generated questions and exam questions via an embedding server.

Talks to any OpenAI-compatible /v1/embeddings endpoint (e.g. llama-server started
with --embedding). Used to build training data (hard distractors, decontamination) and to compare eval-time retrievers.

Questions (generated and exam) are embedded as queries, chunks as documents, in the
format of --prompt-style:
    qwen3  query "Instruct: <task>\nQuery:<q>", document raw text (Qwen3-Embedding)
    gemma  query "task: search result | query: <q>", document "title: none | text: <t>" (EmbeddingGemma)
--max-tokens N truncates inputs longer than N tokens (minus a small margin for special
tokens) with the server's own /tokenize and /detokenize, for models whose context is
shorter than the longest chunk; meta.json records how many inputs were cut per set.

Usage:
    # sanity check: do sample questions retrieve their own source chunk?
    uv run python3 -I pipeline/embed.py check --data data/model --url http://HOST:8080 --gen raft_heldout_fable
    # embed everything; sets already embedded with the same ids are skipped
    uv run python3 -I pipeline/embed.py run --data data/model --url http://HOST:8080 \
        --model-tag qwen3-emb-8b-f16 --gen raft_train raft_heldout_fable
    # eval-time retriever candidate with a 2k context
    uv run python3 -I pipeline/embed.py run --data data/model --url http://HOST:8080 \
        --model-tag embeddinggemma-300m --prompt-style gemma --max-tokens 2048 --gen raft_heldout_fable

If the server needs a key, set LLAMA_API_KEY in the environment (sent as a Bearer token).

Writes to <data>/embeddings/<model-tag>/:
    chunks.npy, eval_exam.npy, q_<gen dir>.npy   float16, L2-normalized, one row per id
    <name>.ids.json                               row ids (chunk id, exam id, or <chunk_id>#<n>)
    meta.json                                     server model, prompt style/formats, eos flag,
                                                  max tokens, truncation counts, dims
"""
import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request

import numpy as np

API_KEY = os.environ.get("LLAMA_API_KEY", "")


def headers():
    h = {'Content-Type': 'application/json'}
    if API_KEY:
        h['Authorization'] = f'Bearer {API_KEY}'
    return h


INSTRUCTION = 'Given a question about electronics, retrieve passages that answer the question'
EOS = '<|endoftext|>'
SEED = 7
SPECIAL_MARGIN = 8  # tokens left free for BOS/EOS the server may add


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def query_text(q, style):
    if style == 'gemma':
        return f'task: search result | query: {q}'
    return f'Instruct: {INSTRUCTION}\nQuery:{q}'


def doc_text(t, style):
    if style == 'gemma':
        return f'title: none | text: {t}'
    return t


def post_json(url, path, payload, timeout):
    req = urllib.request.Request(url.rstrip('/') + path, data=json.dumps(payload).encode(),
                                 headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'server error {e.code}: {e.read()[:500]!r}') from None


def truncate(args, texts, label):
    """Cut texts longer than args.max_tokens with the server tokenizer; returns (texts, n_cut)."""
    if not args.max_tokens:
        return texts, 0
    limit = args.max_tokens - SPECIAL_MARGIN
    out, n_cut = list(texts), 0
    for i, t in enumerate(texts):
        if len(t) <= limit:  # a token covers at least one character
            continue
        toks = post_json(args.url, '/tokenize', {'content': t, 'add_special': False}, args.timeout)['tokens']
        if len(toks) > limit:
            out[i] = post_json(args.url, '/detokenize', {'tokens': toks[:limit]}, args.timeout)['content']
            n_cut += 1
    if n_cut:
        print(f'  {label}: truncated {n_cut}/{len(texts)} inputs to {limit} tokens')
    return out, n_cut


def post(url, inputs, timeout):
    data = post_json(url, '/v1/embeddings', {'input': inputs, 'model': 'embedding'}, timeout)['data']
    return [d['embedding'] for d in sorted(data, key=lambda d: d['index'])]


def embed(args, texts, ids, label, eos):
    """Embed in length-sorted batches; returns an L2-normalized float32 matrix in input order."""
    if eos:
        texts = [t + EOS for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for b in range(0, len(order), args.batch):
        idx = order[b:b + args.batch]
        try:
            vecs = post(args.url, [texts[i] for i in idx], args.timeout)
        except RuntimeError as e:
            sys.exit(f'{label}: batch with {[ids[i] for i in idx]} failed: {e}\n'
                     'If the input is too long, raise the server context per slot / -ub.')
        for i, v in zip(idx, vecs):
            out[i] = v
        print(f'\r  {label}: {min(b + args.batch, len(order))}/{len(order)}', end='', flush=True)
    print()
    m = np.asarray(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def server_model(url, timeout):
    try:
        req = urllib.request.Request(url.rstrip('/') + '/v1/models', headers=headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [m.get('id') for m in json.load(r).get('data', [])]
    except (urllib.error.URLError, ValueError):
        return None


def gen_questions(data, gen):
    """(ids, questions, oracle chunk ids) for every ok record of a generate.py output dir."""
    ids, qs, oracle = [], [], []
    latest = {r['chunk_id']: r for r in load(os.path.join(data, gen, 'generated.jsonl')) if r['status'] == 'ok'}
    for cid, r in sorted(latest.items()):
        for n, q in enumerate(r['questions']):
            ids.append(f'{cid}#{n}')
            qs.append(q['question'])
            oracle.append(cid)
    return ids, qs, oracle


def cmd_check(args):
    """Embed N sample questions, their source chunks and 300 random chunks; report oracle rank."""
    print('server models:', server_model(args.url, args.timeout))
    chunks = {c['id']: c['text'] for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    _, qs, oracle = gen_questions(args.data, args.gen[0])
    rng = random.Random(SEED)
    pick = rng.sample(range(len(qs)), min(args.n, len(qs)))
    pool = sorted({oracle[i] for i in pick})
    pool += rng.sample(sorted(set(chunks) - set(pool)), 300)
    q_texts, _ = truncate(args, [query_text(qs[i], args.prompt_style) for i in pick], 'questions')
    d_texts, _ = truncate(args, [doc_text(chunks[c], args.prompt_style) for c in pool], 'chunks')
    for eos in ((False, True) if args.prompt_style == 'qwen3' else (False,)):  # EOS test is Qwen3-specific
        Q = embed(args, q_texts, [oracle[i] for i in pick], 'questions', eos)
        D = embed(args, d_texts, pool, 'chunks', eos)
        sims = Q @ D.T
        ranks = [int((sims[k] > sims[k, pool.index(oracle[i])]).sum()) + 1 for k, i in enumerate(pick)]
        print(f'append EOS={eos}: oracle rank 1 for {sum(r == 1 for r in ranks)}/{len(ranks)}, '
              f'top 5 for {sum(r <= 5 for r in ranks)}/{len(ranks)}, median rank {sorted(ranks)[len(ranks) // 2]} '
              f'(pool {len(pool)} chunks)')
    print('Use --append-eos with `run` if it ranks clearly better; otherwise leave it off.')


def cmd_run(args):
    out_dir = os.path.join(args.data, 'embeddings', args.model_tag)
    os.makedirs(out_dir, exist_ok=True)
    style = args.prompt_style
    meta = {'model_tag': args.model_tag, 'server_models': server_model(args.url, args.timeout),
            'prompt_style': style, 'instruction': INSTRUCTION if style == 'qwen3' else None,
            'query_format': query_text('{question}', style), 'doc_format': doc_text('{text}', style),
            'append_eos': args.append_eos, 'max_tokens': args.max_tokens, 'truncated': {}}
    meta_path = os.path.join(out_dir, 'meta.json')
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        old.setdefault('prompt_style', 'qwen3')  # folders written before --prompt-style existed
        old.setdefault('max_tokens', 0)
        if any(old.get(k) != meta[k] for k in ('prompt_style', 'instruction', 'append_eos', 'max_tokens')):
            sys.exit(f'{meta_path} was written with different settings; use a new --model-tag')
        meta['dims'] = old.get('dims')
        meta['truncated'] = old.get('truncated', {})
    chunks = load(os.path.join(args.data, 'chunks_clean.jsonl'))
    sets = [('chunks', [c['id'] for c in chunks], [doc_text(c['text'], style) for c in chunks])]
    exam = [r for r in load(os.path.join(args.data, 'eval_exam.jsonl'))]
    sets.append(('eval_exam', [r['id'] for r in exam], [query_text(r['question'], style) for r in exam]))
    for gen in args.gen:
        ids, qs, _ = gen_questions(args.data, gen)
        sets.append((f'q_{os.path.basename(os.path.normpath(gen))}', ids, [query_text(q, style) for q in qs]))
    for name, ids, texts in sets:
        ids_path = os.path.join(out_dir, f'{name}.ids.json')
        if os.path.exists(ids_path) and json.load(open(ids_path)) == ids:
            print(f'  {name}: {len(ids)} already embedded, skipped')
            continue
        texts, meta['truncated'][name] = truncate(args, texts, name)
        m = embed(args, texts, ids, name, args.append_eos)
        np.save(os.path.join(out_dir, f'{name}.npy'), m.astype(np.float16))
        with open(ids_path, 'w') as f:
            json.dump(ids, f)
        meta['dims'] = m.shape[1]
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {out_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='data/<source> folder')
        p.add_argument('--url', required=True, help='embedding server base URL, e.g. http://localhost:8080')
        p.add_argument('--gen', nargs='+', required=True, help='generate.py output dirs, relative to --data')
        p.add_argument('--batch', type=int, default=8, help='inputs per request')
        p.add_argument('--timeout', type=float, default=600)
        p.add_argument('--prompt-style', choices=('qwen3', 'gemma'), default='qwen3',
                       help='query/document format of the embedding model')
        p.add_argument('--max-tokens', type=int, default=0,
                       help='truncate inputs to this many tokens (server tokenizer); 0 = never')
        if name == 'check':
            p.add_argument('--n', type=int, default=40, help='sample questions')
        else:
            p.add_argument('--model-tag', required=True, help='names the output folder, e.g. qwen3-emb-8b-f16')
            p.add_argument('--append-eos', action='store_true', help=f'append {EOS} to every input')
    args = ap.parse_args()
    {'check': cmd_check, 'run': cmd_run}[args.cmd](args)


if __name__ == '__main__':
    main()
