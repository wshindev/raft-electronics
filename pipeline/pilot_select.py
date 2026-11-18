#!/usr/bin/env python3
"""Step 3 pilot: pick ~50 hard images for the Sonnet vs. Opus comparison.

Usage:
    uv run python3 -I pipeline/pilot_select.py --refs data/model/image_refs.jsonl \
        --meta data/model/image_meta.jsonl --pairs data/model/exam_pairs.jsonl

Output (next to --refs unless --out is given):
    pilot/pilot_images.jsonl   one record per image: image, source, category, reason

Selection is deterministic (fixed seed). Categories:
    exam_pair       question figure + answer figure from the same exam item
    exam_question   extra exam-question figures (answer-leak rule applies)
    exam_answer     extra answer-only figures
    schematic       densest XCircuit drawings (components + distinct labels)
    scope           oscilloscope / spectrum-analyzer / waveform figures
    plot            gnuplot / SPICE plots
    photo           photographs
    math / pcb      a few of the rare kinds
"""
import argparse
import json
import os
import random

SEED = 3
# XCircuit library symbols that draw an instrument screen
SCOPE_COMPONENTS = {'fancy_scope', 'scope_display', 'scope_display2',
                    'oscillograph', 'spectrum_display'}
COUNTS = {'exam_pair': 5, 'exam_question': 5, 'exam_answer': 1, 'schematic': 10,
          'scope': 8, 'plot': 6, 'photo': 7, 'math': 2, 'pcb': 1}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def hardness(m):
    comps = m.get('components') or {}
    return sum(comps.values()) + len(set(m.get('labels') or []))


def number(img):
    return int(img.rsplit('_', 1)[1])


def similar(m, chosen, meta):
    """Skip near-duplicates: images numbered within 3 of a chosen one (usually the
    same passage, e.g. one circuit drawn in three states), or a similar label set."""
    if any(abs(number(m['image']) - number(img)) <= 3 for img in chosen):
        return True
    a = set(m.get('labels') or [])
    if not a:
        return False
    for img in chosen:
        b = set(meta[img].get('labels') or [])
        if b and len(a & b) / len(a | b) > 0.6:
            return True
    return False


def pick(cands, n, chosen, meta, key=None, diverse=True):
    out = []
    for m in (sorted(cands, key=key, reverse=True) if key else cands):
        if len(out) == n:
            break
        if m['image'] in chosen or (diverse and similar(m, chosen, meta)):
            continue
        out.append(m)
        chosen.add(m['image'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--pairs', required=True, help='exam_pairs.jsonl')
    ap.add_argument('--out', help='output dir (default: folder of --refs)')
    args = ap.parse_args()

    rng = random.Random(SEED)
    refs = {r['image']: r for r in load(args.refs)}
    meta = {m['image']: m for m in load(args.meta) if m.get('status') != 'missing_eps'}
    ok = lambda img: img in meta and img in refs
    chosen, rows = set(), []

    def add(ms, category, reason):
        for m in ms:
            rows.append({'image': m['image'], 'source': m['source'], 'category': category,
                         'image_kind': m['image_kind'], 'reason': reason(m)})

    def hard_reason(m):
        return f"{m['image_kind']}, hardness {hardness(m)}"

    # Exam pairs: hardest question figure that has its own answer figure(s)
    pairs = [p for p in load(args.pairs) if p['status'] == 'ok'
             and p['question_images'] and p['answer_images']]
    scored = []
    for p in pairs:
        q = [i for i in p['question_images'] if ok(i) and not refs[i]['in_exam_answer']]
        a = [i for i in p['answer_images'] if ok(i) and not refs[i]['in_exam_question']]
        if q and a:
            scored.append((hardness(meta[q[0]]), p['id'], q[0], a[0]))
    scored.sort(reverse=True)
    n_pairs = 0
    for _, pid, q, a in scored:
        if n_pairs == COUNTS['exam_pair']:
            break
        if q in chosen or a in chosen or similar(meta[q], chosen, meta):
            continue
        chosen.update((q, a))
        add([meta[q]], 'exam_pair', lambda m: f'question figure of {pid}')
        add([meta[a]], 'exam_pair', lambda m: f'answer figure of {pid}')
        n_pairs += 1

    # Extra question figures: one also used in module text, one each of photo /
    # instrument screen / plot (where describing without answering is hardest),
    # then the hardest remaining
    q_imgs = sorted((meta[i] for i, r in refs.items() if ok(i) and r['in_exam_question']),
                    key=lambda m: m['image'])
    add(pick([m for m in q_imgs if refs[m['image']]['in_module']], 1, chosen, meta, hardness),
        'exam_question', lambda m: 'question figure also used in module text')
    q_photo = [m for m in q_imgs if m['image_kind'] == 'photo']
    rng.shuffle(q_photo)
    add(pick(q_photo, 1, chosen, meta), 'exam_question', lambda m: 'question photo')
    add(pick([m for m in q_imgs if SCOPE_COMPONENTS & set(m.get('components') or {})],
             1, chosen, meta, hardness), 'exam_question', lambda m: 'question instrument screen')
    add(pick([m for m in q_imgs if m['image_kind'] == 'plot'], 1, chosen, meta, hardness),
        'exam_question', lambda m: 'question plot')
    add(pick(q_imgs, COUNTS['exam_question'] - 4, chosen, meta, hardness),
        'exam_question', hard_reason)
    a_imgs = [meta[i] for i, r in refs.items() if ok(i) and r['in_exam_answer']]
    add(pick(a_imgs, COUNTS['exam_answer'], chosen, meta, hardness), 'exam_answer', hard_reason)

    # The rest: module figures of each kind
    scope = [m for m in meta.values() if SCOPE_COMPONENTS & set(m.get('components') or {})]
    add(pick(scope, COUNTS['scope'], chosen, meta, hardness), 'scope',
        lambda m: 'instrument screen: ' + ', '.join(sorted(SCOPE_COMPONENTS & set(m['components']))))
    xc = [m for m in meta.values() if m['image_kind'] == 'xcircuit']
    add(pick(xc, COUNTS['schematic'], chosen, meta, hardness), 'schematic', hard_reason)

    plots = sorted((m for m in meta.values() if m['image_kind'] == 'plot'), key=lambda m: m['image'])
    landscape = [m for m in plots if m['landscape']]
    add(pick(landscape, 1, chosen, meta), 'plot', lambda m: 'rotated landscape plot')
    rng.shuffle(plots)
    add(pick(plots, COUNTS['plot'] - 1, chosen, meta), 'plot', lambda m: m['creator'])

    for kind, cat in (('photo', 'photo'), ('math_figure', 'math'), ('pcb', 'pcb')):
        ms = sorted((m for m in meta.values() if m['image_kind'] == kind), key=lambda m: m['image'])
        rng.shuffle(ms)
        add(pick(ms, COUNTS[cat], chosen, meta), cat, lambda m: f'random {kind}')

    out_dir = os.path.join(args.out or os.path.dirname(args.refs), 'pilot')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pilot_images.jsonl')
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    by_cat = {}
    for r in rows:
        by_cat[r['category']] = by_cat.get(r['category'], 0) + 1
    print(f'{len(rows)} images -> {path}')
    print(by_cat)


if __name__ == '__main__':
    main()
