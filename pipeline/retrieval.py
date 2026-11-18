"""Shared retrieval helpers: BM25, rank positions, reciprocal rank fusion.

Used by retrieval_eval.py (retriever comparison) and assemble.py (hybrid distractors), so
training distractors and eval retrieval rank chunks the same way. Scripts import it from
their own directory (python -I leaves the script directory off sys.path).
"""
import re

import numpy as np


def tokens(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def bm25(docs, queries, k1=1.2, b=0.75):
    """Okapi BM25 score matrix, queries x docs (inverted index; no extra dependencies)."""
    lens, post = np.empty(len(docs), np.float32), {}
    for j, d in enumerate(docs):
        counts = {}
        for t in tokens(d):
            counts[t] = counts.get(t, 0) + 1
        lens[j] = sum(counts.values())
        for t, c in counts.items():
            post.setdefault(t, ([], []))
            post[t][0].append(j)
            post[t][1].append(c)
    norm_len = k1 * (1 - b + b * lens / lens.mean())
    S = np.zeros((len(queries), len(docs)), np.float32)
    for i, q in enumerate(queries):
        for t in set(tokens(q)):
            if t not in post:
                continue
            js, cs = np.array(post[t][0]), np.array(post[t][1], np.float32)
            idf = np.log(1 + (len(docs) - len(js) + 0.5) / (len(js) + 0.5))
            S[i, js] += idf * cs * (k1 + 1) / (cs + norm_len[js])
    return S


def positions(S):
    """0-based rank of every column, per row, for a score matrix (higher = better)."""
    order = np.argsort(-S, axis=1, kind='stable')
    pos = np.empty_like(order)
    pos[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])
    return pos


def rrf(*pos, k=60):
    """Reciprocal rank fusion score from 0-based rank matrices."""
    return sum(1 / (k + p + 1) for p in pos)
