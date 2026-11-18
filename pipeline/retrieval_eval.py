#!/usr/bin/env python3
"""Step 07: compare eval-time retrievers by recall@k on generated questions.

Offline. Reads embedding folders written by embed.py and ranks the whole chunk corpus
for every question of one generate.py output set (normally the held-out set).

Two definitions of a hit:
    oracle  the question's own source chunk is in the top k
    quotes  some top-k chunk contains every verbatim quote of the reference reasoning
            (whitespace-normalized), or is the oracle. Near-duplicate chunks (repeated
            boilerplate) answer the question as well as the oracle does; this counts them.

--bm25 adds a lexical BM25 row and, per embedding folder, a hybrid row that fuses the
dense and BM25 rankings with reciprocal rank fusion (score = sum of 1/(rrf_k + rank)).
--dump-candidates N writes the first hybrid's top N per question for rerank.py;
--rerank adds a row per rerank.py output (reranked top N, then the hybrid order).

Usage:
    uv run python3 -I pipeline/retrieval_eval.py --data data/model --gen raft_heldout_fable \\
        --emb data/model/embeddings/qwen3-emb-8b-f16 data/model/embeddings/embeddinggemma-300m [--bm25]

Writes <data>/retrieval_<gen>.md (and prints it).
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


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def load_emb(emb_dir, name):
    m = np.load(os.path.join(emb_dir, f'{name}.npy')).astype(np.float32)
    with open(os.path.join(emb_dir, f'{name}.ids.json')) as f:
        return m, json.load(f)


def questions(data, gen):
    """{question id: (oracle chunk id, normalized quotes, question text)}, ids as in embed.py."""
    latest = {r['chunk_id']: r for r in load(os.path.join(data, gen, 'generated.jsonl')) if r['status'] == 'ok'}
    out = {}
    for cid, r in sorted(latest.items()):
        for n, q in enumerate(r['questions']):
            out[f'{cid}#{n}'] = (cid, [x for x in (norm(s) for s in QUOTE_RE.findall(q['reasoning'])) if x],
                                 q['question'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data/<source> folder')
    ap.add_argument('--gen', required=True, help='generate.py output dir (relative to --data) whose questions are queries')
    ap.add_argument('--emb', nargs='+', required=True, help='embedding folders to compare')
    ap.add_argument('--k', type=int, nargs='+', default=[1, 3, 5, 10, 20])
    ap.add_argument('--bm25', action='store_true', help='add BM25 and dense+BM25 (RRF) rows')
    ap.add_argument('--rrf-k', type=int, default=60, help='RRF constant (default 60)')
    ap.add_argument('--dump-candidates', type=int, default=0, metavar='N',
                    help='write the top N of the first hybrid (needs --bm25) to <data>/rerank/candidates_<gen>.jsonl')
    ap.add_argument('--rerank', nargs='+', default=[],
                    help='rerank.py outputs (<data>/rerank/<tag>/<gen>.jsonl) to add as rows (needs --bm25)')
    args = ap.parse_args()
    if (args.dump_candidates or args.rerank) and not args.bm25:
        ap.error('--dump-candidates and --rerank need --bm25')

    gen = os.path.basename(os.path.normpath(args.gen))
    chunk_text = {c['id']: norm(c['text']) for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    qs = questions(args.data, args.gen)

    rows, notes, equiv, hybrids = [], [], None, {}
    if args.bm25:
        bq, bc = sorted(qs), list(chunk_text)
        B = bm25([chunk_text[c] for c in bc], [qs[q][2] for q in bq])
        bq_i, bc_i = {q: i for i, q in enumerate(bq)}, {c: i for i, c in enumerate(bc)}
        notes.append(f'- `bm25`: k1 1.2, b 0.75, lowercase alphanumeric tokens, no stopwords; '
                     f'hybrids fuse ranks with RRF k={args.rrf_k}')

    def add_row(tag, pos, qids, col):
        r_oracle = np.array([pos[i, col[qs[q][0]]] + 1 for i, q in enumerate(qids)])
        r_quotes = np.array([min(pos[i, col[c]] for c in equiv[q]) + 1 for i, q in enumerate(qids)])
        rows.append((tag, r_oracle, r_quotes))

    for emb_dir in args.emb:
        tag = os.path.basename(os.path.normpath(emb_dir))
        meta = json.load(open(os.path.join(emb_dir, 'meta.json')))
        C, cids = load_emb(emb_dir, 'chunks')
        Q, qids = load_emb(emb_dir, f'q_{gen}')
        if set(qids) != set(qs):
            raise SystemExit(f'{tag}: question ids differ from {args.gen}/generated.jsonl; re-embed')
        if equiv is None:  # chunks that answer each question, by quote containment
            equiv = {}
            for qid, (oracle, quotes, _) in qs.items():
                hit = {c for c, t in chunk_text.items() if quotes and all(x in t for x in quotes)}
                equiv[qid] = hit | {oracle}
        col = {c: i for i, c in enumerate(cids)}
        pos = positions(Q @ C.T)
        add_row(tag, pos, qids, col)
        if args.bm25:
            pos_b = positions(B[np.ix_([bq_i[q] for q in qids], [bc_i[c] for c in cids])])
            if not any(r[0] == 'bm25' for r in rows):
                add_row('bm25', pos_b, qids, col)
            pos_h = positions(rrf(pos, pos_b, k=args.rrf_k))
            hybrids[f'{tag} + bm25 (RRF)'] = (pos_h, qids, cids, col)
            add_row(f'{tag} + bm25 (RRF)', pos_h, qids, col)
        cut = meta.get('truncated', {})
        notes.append(f'- `{tag}`: style {meta.get("prompt_style", "qwen3")}, max tokens {meta.get("max_tokens", 0) or "none"}, '
                     f'dims {meta.get("dims")}, truncated chunks {cut.get("chunks", 0)}, '
                     f'truncated questions {cut.get(f"q_{gen}", 0)}')

    if args.dump_candidates:
        base, (pos_h, qids, cids, _) = next(iter(hybrids.items()))
        out = os.path.join(args.data, 'rerank', f'candidates_{gen}.jsonl')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            for i, q in enumerate(qids):
                top = np.argsort(pos_h[i])[:args.dump_candidates]
                f.write(json.dumps({'qid': q, 'question': qs[q][2], 'oracle': qs[q][0], 'base': base,
                                    'rrf_k': args.rrf_k, 'candidates': [cids[j] for j in top]}) + '\n')
        print(f'wrote {out} (top {args.dump_candidates} of {base})')

    for path in args.rerank:  # reranked candidates first (by score), the rest keep the hybrid order
        recs = {r['qid']: r for r in load(path)}
        meta = json.load(open(os.path.join(os.path.dirname(path), 'meta.json')))
        base = next(iter(recs.values()))['base']
        if base not in hybrids:
            raise SystemExit(f'{path}: base retriever {base!r} not among the hybrids; pass its --emb folder')
        pos_h, qids, cids, col = hybrids[base]
        if set(recs) != set(qids):
            raise SystemExit(f'{path}: covers {len(recs)} of {len(qids)} questions; finish rerank.py run')
        S = -pos_h.astype(np.float64)
        stale = 0
        for i, q in enumerate(qids):
            r = recs[q]
            stale += set(r['candidates']) != {cids[j] for j in np.argsort(pos_h[i])[:len(r['candidates'])]}
            for c, s in zip(r['candidates'], r['scores']):
                S[i, col[c]] = 1e6 + s
        if stale:
            print(f'warning: {path}: {stale} questions have candidates that differ from the current hybrid top N')
        depth = len(next(iter(recs.values()))['candidates'])
        add_row(f'{base} > rerank {meta["model_tag"]} (top {depth})', positions(S), qids, col)
        notes.append(f'- rerank `{meta["model_tag"]}`: server {meta.get("server_models")}, {depth} candidates per question')

    n = len(qs)
    multi = sum(len(v) > 1 for v in equiv.values())
    lines = [f'# Retrieval recall — {gen} ({n} questions, {len(chunk_text)} chunks)', '',
             f'{multi} questions have more than one chunk containing all their quotes (near-duplicates).', '']
    head = ' | '.join(f'@{k}' for k in args.k)
    for label, idx in (('oracle chunk in top k', 1), ('quote-equivalent chunk in top k', 2)):
        lines += [f'## {label}', '', f'| retriever | {head} | MRR |', '|---' * (len(args.k) + 2) + '|']
        for row in rows:
            r = row[idx]
            cells = ' | '.join(f'{(r <= k).mean():.1%}' for k in args.k)
            lines.append(f'| {row[0]} | {cells} | {(1 / r).mean():.3f} |')
        lines.append('')
    lines += ['## Settings', ''] + notes + ['']
    text = '\n'.join(lines)
    out = os.path.join(args.data, f'retrieval_{gen}.md')
    with open(out, 'w') as f:
        f.write(text)
    print(text)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
