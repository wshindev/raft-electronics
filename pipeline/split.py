#!/usr/bin/env python3
"""Step 5: train / held-out split of chunks, and the text-only exam eval set.

Held-out unit is the module (any `locations[].module` grouping a source's parser
writes): neighbouring chunks of one module overlap, so splitting single chunks
would leak. A chunk is held out only if every module it appears in is held out;
chunks shared with any training module stay in training.

Usage:
    uv run python3 -I pipeline/split.py --data data/model

Writes to --data:
    split_chunks.jsonl  {id, source, split: train|heldout, generate: bool, type, modules}
    eval_exam.jsonl     exam items with no figure in question or answer
    split_report.txt
"""
import argparse
import collections
import json
import os
import random

HELDOUT_FRACTION = 0.10
SEED = 20261026
# chunk types Claude writes Q/A from; others are retrieval corpus only
GENERATE_TYPES = {'tutorial', 'case_tutorial', 'derivation', 'intro', 'historical',
                  'programming', 'project'}
MIN_GENERATE_CHARS = 400


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data/<source> folder with step 04 outputs')
    ap.add_argument('--heldout-fraction', type=float, default=HELDOUT_FRACTION)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    d = args.data

    chunks = load(os.path.join(d, 'chunks_clean.jsonl'))
    mods_of = {c['id']: sorted({l['module'] for l in c['locations']}) for c in chunks}
    # only modules with a tutorial can be held out (excludes career/pedagogy/tips modules)
    eligible = sorted({m for c in chunks if c['type'] == 'tutorial' for m in mods_of[c['id']]})
    rng = random.Random(args.seed)
    heldout = set(rng.sample(eligible, round(len(eligible) * args.heldout_fraction)))

    rows = []
    for c in chunks:
        ms = mods_of[c['id']]
        split = 'heldout' if all(m in heldout for m in ms) else 'train'
        gen = c['type'] in GENERATE_TYPES and c['n_chars'] >= MIN_GENERATE_CHARS
        rows.append({'id': c['id'], 'source': c['source'], 'split': split, 'generate': gen,
                     'type': c['type'], 'n_chars': c['n_chars'], 'modules': ms})
    with open(os.path.join(d, 'split_chunks.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    pairs = load(os.path.join(d, 'exam_pairs_clean.jsonl'))
    exam = [x for x in pairs if x['status'] == 'ok'
            and not x['question_images'] and not x['answer_images']]
    with open(os.path.join(d, 'eval_exam.jsonl'), 'w') as f:
        for x in exam:
            f.write(json.dumps(x) + '\n')

    tab = collections.Counter((r['split'], r['generate']) for r in rows)
    chars = collections.Counter()
    for r in rows:
        chars[(r['split'], r['generate'])] += r['n_chars']
    shared_kept = sum(1 for r in rows if r['split'] == 'train'
                      and any(m in heldout for m in r['modules']))
    rep = [f'seed {args.seed}; {len(heldout)} of {len(eligible)} eligible modules held out:',
           '  ' + ', '.join(sorted(heldout)),
           f'{"split":<8} {"generate":<9} {"chunks":>6} {"chars":>11}']
    for k in sorted(tab):
        rep.append(f'{k[0]:<8} {str(k[1]):<9} {tab[k]:>6} {chars[k]:>11,}')
    rep += [f'chunks of held-out modules kept in train because shared with a training module: {shared_kept}',
            f'exam eval items (no figures): {len(exam)} in '
            f'{len({x["variant_group"] for x in exam})} variant groups']
    with open(os.path.join(d, 'split_report.txt'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
