#!/usr/bin/env python3
"""Step 07: build the eval prompt sets. Offline.

Two eval sets: the generated held-out questions (--gen, e.g. raft_heldout_fable; each has an
oracle chunk and a reference answer) and the exam items (eval_exam.jsonl). Two conditions:
    nodocs  the question alone
    top<K>  the eval retriever's top K chunks: dense + BM25 over the full corpus fused by
            RRF, top --depth reranked by rerank.py, best first

  candidates  write <data>/rerank/candidates_<set>.jsonl (hybrid top --depth, rerank.py's
              input format) for each set that lacks one; an existing file is checked
              against a fresh ranking instead (same code path as retrieval_eval.py).
  build       read the rerank.py outputs and write the prompt files. --demote-type moves
              candidates of a chunk type (e.g. question) below all others before taking
              the top K (retrieval test).

Usage:
    uv run python3 -I pipeline/eval_sets.py candidates --data data/model --gen raft_heldout_fable \\
        --emb data/model/embeddings/qwen3-emb-8b-f16
    uv run python3 -I pipeline/rerank.py run --data data/model --url http://HOST:8080 \\
        --cands data/model/rerank/candidates_eval_exam.jsonl --model-tag elec-rerank-8B-q8_0
    uv run python3 -I pipeline/eval_sets.py build --data data/model --gen raft_heldout_fable \\
        --rerank data/model/rerank/elec-rerank-8B-q8_0 --out data/model/eval_sets

Writes to --out: <set>_<condition>.jsonl ({id, set, condition, source, messages, doc_ids,
reference, ...} per item; messages hold the system and user turns only) and report.md.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import bm25, positions, rrf  # noqa: E402

QUOTE_RE = re.compile(r'##begin_quote##(.*?)##end_quote##', re.S)
# the training system prompt (assemble.py), and its counterpart without documents
SYSTEM_DOCS = ('Answer the question using the documents provided. Some documents may be '
               'irrelevant. Reason step by step, quoting the supporting text from the documents '
               'as ##begin_quote##...##end_quote##, then give the final answer after "<ANSWER>:".')
SYSTEM_NODOCS = 'Answer the question. Reason step by step, then give the final answer after "<ANSWER>:".'
CHARS_PER_TOKEN = 3.9  # Gemma tokenizer on this corpus, for the length report only


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def load_emb(emb_dir, name):
    m = np.load(os.path.join(emb_dir, f'{name}.npy')).astype(np.float32)
    with open(os.path.join(emb_dir, f'{name}.ids.json')) as f:
        return m, json.load(f)


def eval_sets(data, gen):
    """{set name: [item]}; items carry id, question, oracle (or None) and reference fields."""
    latest = {r['chunk_id']: r for r in load(os.path.join(data, gen, 'generated.jsonl')) if r['status'] == 'ok'}
    g = []
    for cid, r in sorted(latest.items()):
        for n, q in enumerate(r['questions']):
            g.append({'id': f'{cid}#{n}', 'question': q['question'], 'oracle': cid, 'chunk_id': cid,
                      'answer_type': q['answer_type'], 'generator': r['model'],
                      'reference': {'reasoning': q['reasoning'], 'final_answer': q['final_answer']}})
    exam = [{'id': e['id'], 'question': e['question'], 'oracle': None, 'variant_group': e['variant_group'],
             'reference': {'answer': e['answer']}}
            for e in load(os.path.join(data, 'eval_exam.jsonl')) if e['status'] == 'ok']
    return {gen: g, 'eval_exam': exam}


def cmd_candidates(args, chunks, sets):
    tag = os.path.basename(os.path.normpath(args.emb))
    base = f'{tag} + bm25 (RRF)'  # the name retrieval_eval.py gives this hybrid
    C, cids = load_emb(args.emb, 'chunks')
    texts = [norm(chunks[c]['text']) for c in cids]
    for name, items in sets.items():
        Q, qids = load_emb(args.emb, name if name == 'eval_exam' else f'q_{name}')
        by_id = {it['id']: it for it in items}
        if set(qids) != set(by_id):
            sys.exit(f'{name}: embedding ids differ from the eval items; re-embed')
        pos = positions(rrf(positions(Q @ C.T), positions(bm25(texts, [by_id[q]['question'] for q in qids])),
                            k=args.rrf_k))
        recs = [{'qid': q, 'question': by_id[q]['question'], 'oracle': by_id[q]['oracle'], 'base': base,
                 'rrf_k': args.rrf_k, 'candidates': [cids[j] for j in np.argsort(pos[i])[:args.depth]]}
                for i, q in enumerate(qids)]
        out = os.path.join(args.data, 'rerank', f'candidates_{name}.jsonl')
        if os.path.exists(out):
            old = {r['qid']: r for r in load(out)}
            diff = sum(old.get(r['qid'], {}).get('candidates') != r['candidates'] for r in recs)
            print(f'{out} exists: {diff} of {len(recs)} questions differ from a fresh ranking'
                  + ('' if diff else ' (consistent)'))
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        print(f'wrote {out} ({len(recs)} questions, top {args.depth} of {base})')


def cmd_build(args, chunks, sets):
    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(os.path.join(args.rerank, 'meta.json')))
    demote = set(args.demote_type or ())
    lines = [f'# Eval prompt sets', '', f'Retrieval: {{base}} top {{depth}} → rerank `{meta["model_tag"]}` → top {args.top}, '
             'best first. Documents are the clean chunk text, formatted as in training (assemble.py).', '']
    if demote:
        lines += [f'Demoted chunk types (ranked below all other candidates): {", ".join(sorted(demote))}.', '']
    lines += ['| file | items | est. prompt tokens median | max | oracle in docs |', '|---|---|---|---|---|']
    base = depth = None
    for name, items in sets.items():
        path = os.path.join(args.rerank, f'{name}.jsonl')
        if not os.path.exists(path):
            sys.exit(f'{path} missing: run rerank.py on {args.data}/rerank/candidates_{name}.jsonl first')
        rr = {r['qid']: r for r in load(path)}
        if set(rr) != {it['id'] for it in items}:
            sys.exit(f'{path}: covers {len(rr)} of {len(items)} items; finish the rerank.py run')
        base, depth = next(iter(rr.values()))['base'], len(next(iter(rr.values()))['candidates'])
        short = 'exam' if name == 'eval_exam' else name.removeprefix('raft_heldout_')
        for cond in ('nodocs', f'top{args.top}'):
            rows = []
            for it in items:
                if cond == 'nodocs':
                    docs, msgs = [], [{'role': 'system', 'content': SYSTEM_NODOCS},
                                      {'role': 'user', 'content': f'Question: {it["question"]}'}]
                else:
                    r = rr[it['id']]
                    order = sorted(range(len(r['candidates'])),  # stable on ties
                                   key=lambda i: (chunks[r['candidates'][i]]['type'] in demote, -r['scores'][i]))
                    docs = [r['candidates'][i] for i in order[:args.top]]
                    body = '\n\n'.join(f'<DOCUMENT>\n{chunks[d]["text"]}\n</DOCUMENT>' for d in docs)
                    msgs = [{'role': 'system', 'content': SYSTEM_DOCS},
                            {'role': 'user', 'content': f'{body}\n\nQuestion: {it["question"]}'}]
                row = {'id': it['id'], 'set': short, 'condition': cond, 'source': args.source, 'messages': msgs,
                       'doc_ids': docs, **{k: v for k, v in it.items() if k not in ('id', 'question', 'oracle')}}
                if it['oracle'] and docs:
                    row['oracle_present'] = it['oracle'] in docs
                rows.append(row)
            fname = f'{short}_{cond}.jsonl'
            with open(os.path.join(args.out, fname), 'w') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            est = sorted(sum(len(m['content']) for m in row['messages']) / CHARS_PER_TOKEN for row in rows)
            hit = sum(row.get('oracle_present', False) for row in rows) if items[0]['oracle'] and cond != 'nodocs' else None
            lines.append(f'| {fname} | {len(rows)} | {est[len(est) // 2]:,.0f} | {est[-1]:,.0f} | '
                         f'{"—" if hit is None else f"{hit} ({hit / len(rows):.1%})"} |')
    lines[2] = lines[2].format(base=base, depth=depth)
    lines += ['', f'System prompt with documents (training): {SYSTEM_DOCS}', '',
              f'System prompt without documents: {SYSTEM_NODOCS}', '',
              f'Token estimates use {CHARS_PER_TOKEN} chars/token (Gemma); the server context per slot '
              'must hold the longest prompt plus the max output tokens.', '']
    text = '\n'.join(lines)
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('candidates', 'build'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='data/<source> folder')
        p.add_argument('--gen', required=True, help='generate.py output dir of the held-out questions, relative to --data')
        if name == 'candidates':
            p.add_argument('--emb', required=True, help='embedding folder of the eval retriever (dense stage)')
            p.add_argument('--depth', type=int, default=20, help='hybrid candidates passed to the reranker')
            p.add_argument('--rrf-k', type=int, default=60)
        else:
            p.add_argument('--rerank', required=True, help='rerank.py output folder (<data>/rerank/<model-tag>)')
            p.add_argument('--top', type=int, default=5, help='documents per prompt (training used 5)')
            p.add_argument('--demote-type', action='append', metavar='TYPE',
                           help='chunk type to rank below all other candidates (repeatable)')
            p.add_argument('--out', required=True)
    args = ap.parse_args()
    chunks = {c['id']: c for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    args.source = next(iter(chunks.values()))['source']
    sets = eval_sets(args.data, args.gen)
    {'candidates': cmd_candidates, 'build': cmd_build}[args.cmd](args, chunks, sets)


if __name__ == '__main__':
    main()
