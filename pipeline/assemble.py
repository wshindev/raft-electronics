#!/usr/bin/env python3
"""Step 6b-2: decontaminate generated Q/A and assemble RAFT training examples.

Inputs (all under --data): chunks_clean.jsonl, split_chunks.jsonl, eval_exam.jsonl,
<gen>/generated.jsonl from generate.py, and embeddings from embed.py.

1. Quality filter: drop questions with no quote or any non-verbatim quote.
2. Decontamination: drop questions whose embedding is close to
   an exam question (cosine >= --decon-sim) or that share a --ngram word run with
   an exam question or answer.
3. Hard distractors: training-split chunks ranked by similarity to the
   question; the oracle, near-duplicates of it, chunks containing the answer's
   quotes and chunks containing exam text are excluded. Sampled from ranks
   --rank-lo..--rank-hi, plus random ones. With --bm25 the ranking is the eval
   retriever's first stage: dense and BM25 ranks over the full
   corpus fused by RRF, then restricted to the pool.
4. Assembly: oracle present with probability --p-oracle, shuffled documents,
   chat-format messages. A dev slice is split off by module.

Usage:
    uv run python3 -I pipeline/assemble.py --data data/model --gen raft_train \
        --emb data/model/embeddings/qwen3-emb-8b-f16 --out data/model/sft_sonnet [--bm25]

Writes to --out: train.jsonl, dev.jsonl, decon_report.md, assemble_report.txt
"""
import argparse
import collections
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import bm25, positions, rrf  # noqa: E402

SEED = 20261027
QUOTE_RE = re.compile(r'##begin_quote##(.*?)##end_quote##', re.S)
SYSTEM = ('Answer the question using the documents provided. Some documents may be '
          'irrelevant. Reason step by step, quoting the supporting text from the documents '
          'as ##begin_quote##...##end_quote##, then give the final answer after "<ANSWER>:".')


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def norm_ws(s):
    return re.sub(r'\s+', ' ', s).strip()


def words(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def ngrams(s, n):
    w = words(s)
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_emb(emb_dir, name):
    m = np.load(os.path.join(emb_dir, f'{name}.npy')).astype(np.float32)
    with open(os.path.join(emb_dir, f'{name}.ids.json')) as f:
        return m, json.load(f)


def load_questions(data, gen):
    """Questions in embed.py's order: latest ok record per chunk, sorted by chunk id."""
    latest = {r['chunk_id']: r for r in load(os.path.join(data, gen, 'generated.jsonl')) if r['status'] == 'ok'}
    out = []
    for cid, r in sorted(latest.items()):
        for n, q in enumerate(r['questions']):
            out.append({'id': f'{cid}#{n}', 'chunk_id': cid, 'model': r['model'], **q})
    return out


def format_example(q, docs, chunks, oracle_present):
    body = '\n\n'.join(f'<DOCUMENT>\n{chunks[d]["text"]}\n</DOCUMENT>' for d in docs)
    return {
        'id': q['id'], 'source': chunks[q['chunk_id']]['source'], 'chunk_id': q['chunk_id'],
        'generator': q['model'], 'answer_type': q['answer_type'],
        'oracle_present': oracle_present, 'doc_ids': docs,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'{body}\n\nQuestion: {q["question"]}'},
            {'role': 'assistant', 'content': f'{q["reasoning"]}\n\n<ANSWER>: {q["final_answer"]}'},
        ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data/<source> folder')
    ap.add_argument('--gen', required=True, help='generate.py output dir (training split), relative to --data')
    ap.add_argument('--emb', required=True, help='embed.py output folder')
    ap.add_argument('--out', required=True)
    ap.add_argument('--decon-sim', type=float, default=0.80, help='drop if cosine to an exam question >= this')
    ap.add_argument('--ngram', type=int, default=13, help='drop if sharing an n-word run with exam text')
    ap.add_argument('--dup-sim', type=float, default=0.95, help='chunks this similar to the oracle are not distractors')
    ap.add_argument('--k', type=int, default=4, help='distractors per example')
    ap.add_argument('--hard', type=int, default=3, help='of which hard (the rest random)')
    ap.add_argument('--rank-lo', type=int, default=3)
    ap.add_argument('--rank-hi', type=int, default=20)
    ap.add_argument('--bm25', action='store_true', help='rank distractors by dense + BM25 (RRF) instead of dense only')
    ap.add_argument('--rrf-k', type=int, default=60, help='RRF constant for --bm25')
    ap.add_argument('--p-oracle', type=float, default=0.8)
    ap.add_argument('--dev-frac', type=float, default=0.05, help='fraction of modules held back as dev')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    chunks = {c['id']: c for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    split = {r['id']: r for r in load(os.path.join(args.data, 'split_chunks.jsonl'))}
    exam = load(os.path.join(args.data, 'eval_exam.jsonl'))
    qs = load_questions(args.data, args.gen)
    gen_name = os.path.basename(os.path.normpath(args.gen))
    Q, qids = load_emb(args.emb, f'q_{gen_name}')
    assert qids == [q['id'] for q in qs], 'embeddings are out of date for this generated.jsonl; rerun embed.py'
    C, cids = load_emb(args.emb, 'chunks')
    E, eids = load_emb(args.emb, 'eval_exam')
    cpos = {c: i for i, c in enumerate(cids)}
    assert all(split[q['chunk_id']]['split'] == 'train' for q in qs), 'training generator dir contains held-out chunks'
    stats = collections.Counter(questions=len(qs))

    # 1. quality filter
    keep = []
    for i, q in enumerate(qs):
        text = norm_ws(chunks[q['chunk_id']]['text'])
        quotes = [norm_ws(x) for x in QUOTE_RE.findall(q['reasoning'])]
        if not quotes:
            stats['drop_no_quote'] += 1
        elif not all(x and x in text for x in quotes):
            stats['drop_bad_quote'] += 1
        else:
            q['quotes'] = quotes
            keep.append(i)

    # 2. decontamination against exam items
    exam_grams = set()
    for e in exam:
        exam_grams |= ngrams(e['question'], args.ngram) | ngrams(e['answer'], args.ngram)
    S = Q[keep] @ E.T
    best, arg = S.max(1), S.argmax(1)
    exam_text = {e['id']: e for e in exam}
    kept, report = [], []
    for j, i in enumerate(keep):
        q = qs[i]
        hit_gram = bool(ngrams(q['question'] + ' ' + q['final_answer'], args.ngram) & exam_grams)
        if best[j] >= args.decon_sim or hit_gram:
            stats['drop_decon_sim' if best[j] >= args.decon_sim else 'drop_decon_ngram'] += 1
            report.append((float(best[j]), hit_gram, q, exam_text[eids[arg[j]]]))
        else:
            kept.append(i)
            if best[j] >= args.decon_sim - 0.05:
                report.append((float(best[j]), False, q, exam_text[eids[arg[j]]]))
    report.sort(key=lambda r: -r[0])
    with open(os.path.join(args.out, 'decon_report.md'), 'w') as f:
        f.write(f'# Decontamination — {gen_name}, threshold {args.decon_sim}, {args.ngram}-gram\n\n'
                f'Dropped and borderline ({args.decon_sim - 0.05:.2f}–{args.decon_sim}, kept) pairs, most similar first.\n\n')
        for sim, gram, q, e in report:
            verdict = 'DROP' if sim >= args.decon_sim or gram else 'kept'
            f.write(f'**{verdict}** sim {sim:.3f}{" + n-gram" if gram else ""} — `{q["id"]}` vs `{e["id"]}`\n\n'
                    f'- GEN: {q["question"]}\n- EXAM: {norm_ws(e["question"])[:400]}\n\n')

    # 3. distractor pool: training-split chunks without exam text
    pool = [c for c in cids if split[c]['split'] == 'train'
            and not (ngrams(chunks[c]['text'], args.ngram) & exam_grams)]
    stats['pool_chunks'] = len(pool)
    stats['pool_excluded_exam_text'] = sum(1 for c in cids if split[c]['split'] == 'train') - len(pool)
    P = C[[cpos[c] for c in pool]]
    pool_text = [norm_ws(chunks[c]['text']) for c in pool]

    # 4. assembly, dev slice by module
    # drawn from the split, not from surviving questions, so every generator gets the same dev modules
    modules = sorted({r['modules'][0] for r in split.values() if r['split'] == 'train' and r['generate']})
    dev_mods = set(random.Random(args.seed).sample(modules, max(1, round(len(modules) * args.dev_frac))))
    out = {'train': [], 'dev': []}
    ranks_used = []
    if args.bm25:  # same ranking as retrieval_eval.py: full corpus, then restricted to the pool
        dense = positions(Q[kept] @ C.T)
        lex = positions(bm25([norm_ws(chunks[c]['text']) for c in cids], [qs[i]['question'] for i in kept]))
        sims_all = rrf(dense, lex, k=args.rrf_k)[:, [cpos[c] for c in pool]]
        del dense, lex
    else:
        sims_all = Q[kept] @ P.T
    for j, i in enumerate(kept):
        q = qs[i]
        oracle = q['chunk_id']
        ov = C[cpos[oracle]]
        order = np.argsort(-sims_all[j])
        cands = []
        for p in order[:args.rank_hi * 3]:
            c = pool[p]
            if c == oracle or float(P[p] @ ov) >= args.dup_sim or any(x in pool_text[p] for x in q['quotes']):
                continue
            cands.append(c)
        window = cands[args.rank_lo - 1:args.rank_hi]
        n_hard = min(args.hard, len(window))
        hard = rng.sample(window, n_hard)
        ranks_used += [cands.index(h) + 1 for h in hard]
        excluded = set(hard) | {oracle} | set(cands[:args.rank_hi])
        present = rng.random() < args.p_oracle
        n_rand = args.k - n_hard + (0 if present else 1)  # oracle-absent examples keep the same document count
        rand = []
        while len(rand) < n_rand:
            c = rng.choice(pool)
            if c not in excluded and c not in rand:
                rand.append(c)
        docs = hard + rand + ([oracle] if present else [])
        rng.shuffle(docs)
        ex = format_example(q, docs, chunks, present)
        out['dev' if split[oracle]['modules'][0] in dev_mods else 'train'].append(ex)

    for name, rows in out.items():
        rng.shuffle(rows)
        with open(os.path.join(args.out, f'{name}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')

    lens = sorted(sum(len(m['content']) for m in r['messages']) for r in out['train'] + out['dev'])
    pct = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    ranker = f'dense + BM25 (RRF k={args.rrf_k})' if args.bm25 else 'dense'
    lines = [f'gen {gen_name}, emb {os.path.basename(os.path.normpath(args.emb))}, distractor ranking {ranker}, seed {args.seed}',
             f'settings: decon_sim {args.decon_sim}, ngram {args.ngram}, dup_sim {args.dup_sim}, k {args.k} '
             f'(hard {args.hard} from ranks {args.rank_lo}-{args.rank_hi}), p_oracle {args.p_oracle}, dev_frac {args.dev_frac}',
             *(f'{k}: {v}' for k, v in stats.items()),
             f'examples: train {len(out["train"])}, dev {len(out["dev"])} ({len(dev_mods)} dev modules)',
             f'oracle present: {sum(r["oracle_present"] for r in out["train"] + out["dev"])}',
             f'hard distractor rank (after exclusions): median {sorted(ranks_used)[len(ranks_used) // 2] if ranks_used else "-"}',
             f'example length, chars (~tokens = chars/3): median {pct(.5):,}, p90 {pct(.9):,}, p99 {pct(.99):,}, max {lens[-1]:,}']
    with open(os.path.join(args.out, 'assemble_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
