#!/usr/bin/env python3
"""Step 08: export the eval set as self-contained files for publishing. Offline.

Joins what the eval runs used into one row per question: question, reference answer, key
points, the eval retriever's top documents, scoring group and provenance. Reads the prompt
files actually run (eval_sets.py output), the key points (keypoints.py) and the clean corpus.
Then rebuilds every prompt from the export and checks it matches the prompt files exactly.

Usage:
    uv run python3 -I pipeline/export_eval.py --data data/model --gen raft_heldout_fable \\
        --sets data/model/eval_sets --keypoints data/model/keypoints --out data/model/eval_release

Writes to --out:
    exam.jsonl, <gen short name>.jsonl   one row per question
    corpus.jsonl                         every retrievable document {id, type, module, module_title, text}
    prompts.json                         system prompts and the user-turn templates
    report.md                            counts and the round-trip check
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sets import SYSTEM_DOCS, SYSTEM_NODOCS, eval_sets, load  # noqa: E402
from judge import GROUP_MERGE  # noqa: E402

USER_NODOCS = 'Question: {question}'
USER_DOCS = '{documents}\n\nQuestion: {question}'
DOC = '<DOCUMENT>\n{text}\n</DOCUMENT>'
DOC_SEP = '\n\n'
POINT_KEYS = ('kind', 'label', 'statement', 'values', 'rel_tol', 'abs_tol', 'accept')


def module_of(chunk):
    loc = (chunk.get('locations') or [{}])[0]
    return loc.get('module'), loc.get('module_title')


def prompt(row, corpus, with_docs):
    if not with_docs:
        return [{'role': 'system', 'content': SYSTEM_NODOCS},
                {'role': 'user', 'content': USER_NODOCS.format(question=row['question'])}]
    docs = DOC_SEP.join(DOC.format(text=corpus[d]['text']) for d in row['retrieved'])
    return [{'role': 'system', 'content': SYSTEM_DOCS},
            {'role': 'user', 'content': USER_DOCS.format(documents=docs, question=row['question'])}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data/<source> folder')
    ap.add_argument('--gen', required=True, help='generate.py output dir of the held-out questions, relative to --data')
    ap.add_argument('--sets', required=True, help='eval_sets.py output folder (the prompt files that were run)')
    ap.add_argument('--keypoints', required=True, help='keypoints.py output folder')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    chunks = {c['id']: c for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    exam_src = {e['id']: e for e in load(os.path.join(args.data, 'eval_exam.jsonl'))}
    kp = {r['id']: r for r in load(os.path.join(args.keypoints, 'keypoints.jsonl'))}
    bad = [i for i, r in kp.items() if r['status'] != 'ok']
    if bad:
        sys.exit(f'{len(bad)} key-point records not ok, e.g. {bad[:3]}')

    os.makedirs(args.out, exist_ok=True)
    lines = ['# Eval set export', '', f'Source: `{args.data}`, prompts `{args.sets}`, key points `{args.keypoints}`.', '',
             '| file | rows | key points (numeric / exact / conceptual) | notes: ungradable / reference issue | '
             'top-5 has source chunk | prompts rebuilt = run |', '|---|---|---|---|---|---|']
    used = set()
    for name, items in eval_sets(args.data, args.gen).items():
        short = 'exam' if name == 'eval_exam' else name.removeprefix('raft_heldout_')
        runs = {}
        for path in sorted(os.listdir(args.sets)):
            if path.startswith(f'{short}_') and path.endswith('.jsonl'):
                runs[path[len(short) + 1:-6]] = {r['id']: r for r in load(os.path.join(args.sets, path))}
        top = [c for c in runs if c != 'nodocs']
        if 'nodocs' not in runs or len(top) != 1:
            sys.exit(f'{args.sets}: expected {short}_nodocs.jsonl and one {short}_top<K>.jsonl, found {sorted(runs)}')
        top = runs[top[0]]
        rows = []
        for it in items:
            k = kp.get(it['id'])
            if k is None or it['id'] not in top or it['id'] not in runs['nodocs']:
                sys.exit(f'{it["id"]}: missing from the key points or a prompt file')
            row = {'id': it['id'], 'source': next(iter(chunks.values()))['source'], 'question': it['question']}
            if it['oracle']:
                mod, title = module_of(chunks[it['chunk_id']])
                row |= {'answer_type': it['answer_type'], 'reference': it['reference'],
                        'source_chunk': it['chunk_id'], 'module': mod, 'module_title': title,
                        'score_group': it['chunk_id']}
            else:
                e = exam_src[it['id']]
                vg = it['variant_group']
                row |= {'reference': it['reference'], 'variant_group': vg, 'score_group': GROUP_MERGE.get(vg, vg),
                        'topics': sorted({p['topic'] for p in e.get('pools', []) if p.get('topic')})}
            row['key_points'] = [{key: p[key] for key in POINT_KEYS if key in p} for p in k['points']]
            notes = {n: k[n] for n in ('ungradable', 'reference_issue') if k.get(n)}
            if notes:
                row['notes'] = notes
            row['retrieved'] = top[it['id']]['doc_ids']
            if it['oracle']:
                row['source_chunk_retrieved'] = it['chunk_id'] in row['retrieved']
            row['provenance'] = {'question': 'source exam' if not it['oracle'] else it['generator'],
                                 'reference': 'source exam' if not it['oracle'] else it['generator'],
                                 'key_points': k['model'], 'key_points_prompt': k['prompt_version']}
            used.update(row['retrieved'])
            rows.append(row)

        same = sum(prompt(r, chunks, False) == runs['nodocs'][r['id']]['messages'] and
                   prompt(r, chunks, True) == top[r['id']]['messages'] for r in rows)
        if same != len(rows):
            sys.exit(f'{short}: only {same} of {len(rows)} rebuilt prompts match the prompt files')
        with open(os.path.join(args.out, f'{short}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        kinds = collections.Counter(p['kind'] for r in rows for p in r['key_points'])
        nn = collections.Counter(n for r in rows for n in r.get('notes', {}))
        hit = sum(r.get('source_chunk_retrieved', False) for r in rows)
        lines.append(f'| {short}.jsonl | {len(rows)} | {sum(kinds.values())} ({kinds["numeric"]} / {kinds["exact"]} / '
                     f'{kinds["conceptual"]}) | {nn["ungradable"]} / {nn["reference_issue"]} | '
                     f'{f"{hit} ({hit / len(rows):.1%})" if rows[0].get("source_chunk") else "—"} | {same} / {len(rows)} |')

    with open(os.path.join(args.out, 'corpus.jsonl'), 'w') as f:
        for c in chunks.values():
            mod, title = module_of(c)
            f.write(json.dumps({'id': c['id'], 'type': c['type'], 'module': mod, 'module_title': title,
                                'text': c['text']}, ensure_ascii=False) + '\n')
    with open(os.path.join(args.out, 'prompts.json'), 'w') as f:
        json.dump({'system_with_documents': SYSTEM_DOCS, 'system_without_documents': SYSTEM_NODOCS,
                   'user_with_documents': USER_DOCS, 'user_without_documents': USER_NODOCS,
                   'document': DOC, 'document_separator': DOC_SEP,
                   'documents_order': 'retrieved, best first'}, f, indent=2, ensure_ascii=False)
    lines += ['', f'corpus.jsonl: {len(chunks)} documents ({len(used)} appear in a top 5).', '',
              'Prompts rebuilt from the export (prompts.json + corpus.jsonl) are compared with the prompt files '
              'that were run, for both conditions; the export stops if any differ.', '']
    text = '\n'.join(lines)
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
