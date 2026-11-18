#!/usr/bin/env python3
"""Step 07: score eval runs. Offline.

  extract  pull the final answer out of every eval_run.py output and report format stats
           per config and file. The answer is the text after the last "<ANSWER>" marker
           (colon optional, markdown bold tolerated), up to a closing "</ANSWER>" if the
           model wrote a tag pair; a last marker with nothing after it is read as a closing
           tag without the slash, and the one before it is used. If every marker is empty (the
           model wrote its answer first and closed with a bare "<ANSWER>", common with thinking
           on), it first tries a "Final Answer" heading line, then takes the whole output before the
           last marker ("before_marker"). Without a
           marker: the text after a "Final Answer" heading line, else the last TAIL_CHARS characters.
  ground   quote grounding for every run with documents: is each ##begin_quote## span in the
           documents the model was given? Documents are read from the eval prompt itself
           (<DOCUMENT> blocks, in doc_ids order). A quote is "exact" if, after normalizing
           (case, whitespace, markdown * and `, $, curly quotes), every segment between
           ellipses is a substring of one document; "partial" if its longest common run with a
           document is >= PARTIAL_FRAC of its length and >= PARTIAL_MIN chars; else "none".
           Where the item has a source chunk (chunk_id, e.g. Fable), it also records whether
           a quote came from it. Quotes not in the documents but in the question text are
           counted as "question" (copied from the prompt, not evidence).

Usage:
    uv run python3 -I pipeline/score.py extract --runs data/model/eval_runs --out data/model/scores \\
        [--sets data/model/eval_sets] [--configs e4b_base e4b_sonnet]
    uv run python3 -I pipeline/score.py ground --runs data/model/eval_runs --sets data/model/eval_sets \\
        --out data/model/scores [--configs ...]

Writes to --out: answers/<config>/<file>.jsonl ({id, set, condition, answer, method, closed_tag,
n_markers, quotes, finish_reason, completion_tokens} per item) and format_report.md.
ground (needs --sets) writes grounding/<config>/<file>.jsonl and grounding_report.md.
Config folders starting with "_" (e.g. _pilot_max1024) are skipped unless named in --configs.
"""
import argparse
import json
import os
import re
import statistics
import sys
from difflib import SequenceMatcher

OPEN_RE = re.compile(r'(?:\*\*)?<\s*ANSWER\s*>(?:\*\*)?[ \t]*(:)?(?:\*\*)?', re.I)  # not "</ANSWER>"
CLOSE_RE = re.compile(r'<\s*/\s*ANSWER\s*>', re.I)
HEADING_RE = re.compile(r'^[#*\s]*final answer\b[^\n]*$', re.I | re.M)
QUOTE = '##begin_quote##'
TAIL_CHARS = 1500
METHODS = ('colon', 'no_colon', 'before_marker', 'heading', 'tail')
QUOTE_RE = re.compile(r'##begin_quote##(.*?)##end_quote##', re.S)
DOC_RE = re.compile(r'<DOCUMENT>\n(.*?)\n</DOCUMENT>', re.S)
ELLIPSIS_RE = re.compile(r'\.\.\.|\u2026|\[\.\.\.\]')
PARTIAL_FRAC = 0.6
PARTIAL_MIN = 20
MATCHES = ('exact', 'partial', 'question', 'none')


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract(output):
    """(answer, method, closed_tag, n_markers) for one model output."""
    opens = list(OPEN_RE.finditer(output))
    # last marker first; an empty one is taken as a closing tag written without the slash
    for i in range(len(opens) - 1, -1, -1):
        m = opens[i]
        rest = output[m.end():opens[i + 1].start() if i + 1 < len(opens) else len(output)]
        close = CLOSE_RE.search(rest)
        answer = (rest[:close.start()] if close else rest).strip().strip('*').strip()
        if answer:
            closed = bool(close) or i + 1 < len(opens)
            return answer, 'colon' if m.group(1) else 'no_colon', closed, len(opens)
    heads = list(HEADING_RE.finditer(output))
    if heads:
        h = heads[-1]
        line = h.group(0)
        after = (line.split(':', 1)[1] if ':' in line else '') + output[h.end():]
        answer = after.strip().strip('*').strip()
        if answer:
            return answer, 'heading', False, len(opens)
    if opens:  # only empty markers: the answer was written before them
        answer = output[:opens[-1].start()].strip()
        if answer:
            return answer, 'before_marker', False, len(opens)
    return output[-TAIL_CHARS:].strip(), 'tail', False, len(opens)


def median(xs):
    return statistics.median(xs) if xs else 0


def cmd_extract(args):
    configs = args.configs or sorted(d for d in os.listdir(args.runs)
                                     if os.path.exists(os.path.join(args.runs, d, 'meta.json')) and not d.startswith('_'))
    lines = ['# Answer extraction and format stats', '',
             f'Answer = text after the last `<ANSWER>` marker (colon optional), up to `</ANSWER>` if present; '
             'else after a "Final Answer" heading; else, if every marker is empty, the whole output before the '
             f'last one ("before marker"); else the last {TAIL_CHARS} characters ("tail"). '
             '"cut" = stopped at max tokens.', '',
             '| config | file | items | cut | colon | no colon | tag pair | before marker | heading | tail | cut + tail | '
             '>1 marker | quotes docs | answer chars median | output tokens median | max |',
             '|' + '---|' * 16]
    for config in configs:
        cdir = os.path.join(args.runs, config)
        meta = json.load(open(os.path.join(cdir, 'meta.json')))
        files = sorted(f for f in os.listdir(cdir) if f.endswith('.jsonl'))
        odir = os.path.join(args.out, 'answers', config)
        os.makedirs(odir, exist_ok=True)
        for fname in files:
            runs = load(os.path.join(cdir, fname))
            if args.sets:
                want = {r['id'] for r in load(os.path.join(args.sets, fname))}
                got = {r['id'] for r in runs}
                if want != got:
                    print(f'WARNING {config}/{fname}: {len(want - got)} items missing, {len(got - want)} unknown',
                          file=sys.stderr)
            rows = []
            for r in runs:
                answer, method, closed, n = extract(r['output'])
                rows.append({'id': r['id'], 'set': r['set'], 'condition': r['condition'], 'answer': answer,
                             'method': method, 'closed_tag': closed, 'n_markers': n,
                             'quotes': r['output'].count(QUOTE), 'finish_reason': r['finish_reason'],
                             'completion_tokens': r['completion_tokens']})
            with open(os.path.join(odir, fname), 'w') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            cnt = {m: sum(row['method'] == m for row in rows) for m in METHODS}
            cut = [row for row in rows if row['finish_reason'] == 'length']
            docs = 'nodocs' not in fname
            toks = [row['completion_tokens'] or 0 for row in rows]
            lines.append(
                f'| {config} | {fname.removesuffix(".jsonl")} | {len(rows)} | {len(cut)} | {cnt["colon"]} | '
                f'{cnt["no_colon"]} | {sum(row["closed_tag"] for row in rows)} | {cnt["before_marker"]} | {cnt["heading"]} | {cnt["tail"]} | '
                f'{sum(row["method"] == "tail" for row in cut)} | {sum(row["n_markers"] > 1 for row in rows)} | '
                f'{sum(row["quotes"] > 0 for row in rows) if docs else "—"} | '
                f'{median([len(row["answer"]) for row in rows]):,.0f} | {median(toks):,.0f} | {max(toks, default=0):,} |')
        lines.append(f'|  | max_tokens {meta["params"]["max_tokens"]}, LoRA: '
                     f'{meta.get("lora_adapters", "not recorded")} |' + ' |' * 14)
    text = '\n'.join(lines) + '\n'
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'format_report.md'), 'w') as f:
        f.write(text)
    print(text)


def norm(text):
    text = text.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    text = re.sub(r'[*`$]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def ground_quote(quote, docs):
    """(match, doc index or None) for one quote against the normalized documents."""
    segs = [s for s in (norm(x) for x in ELLIPSIS_RE.split(quote)) if s]
    if not segs:
        return 'none', None
    for i, d in enumerate(docs):
        if all(s in d for s in segs):
            return 'exact', i
    q = ' '.join(segs)
    best, best_i = 0, None
    for i, d in enumerate(docs):
        m = SequenceMatcher(None, q, d, autojunk=False).find_longest_match(0, len(q), 0, len(d))
        if m.size > best:
            best, best_i = m.size, i
    if best >= PARTIAL_MIN and best >= PARTIAL_FRAC * len(q):
        return 'partial', best_i
    return 'none', None


def pct(a, b):
    return f'{a / b:.0%}' if b else '—'


def cmd_ground(args):
    configs = args.configs or sorted(d for d in os.listdir(args.runs)
                                     if os.path.exists(os.path.join(args.runs, d, 'meta.json')) and not d.startswith('_'))
    lines = ['# Quote grounding', '',
             'Quotes = `##begin_quote##` spans in runs with documents. **exact**: every segment between ellipses is '
             'in one of the documents given (after normalizing case, whitespace, `*`, backticks, `$`, curly quotes). '
             f'**partial**: longest common run with a document >= {PARTIAL_FRAC:.0%} of the quote and >= '
             f'{PARTIAL_MIN} chars. **question**: not in the documents but in the question text (exact or partial). **none**: none of these. "items all grounded" = items with quotes and no "none" quote. "source quoted" = items whose source chunk was among the documents and was '
             'quoted (Fable only; exam items have no source chunk).', '',
             '| config | file | items | items with quotes | quotes | exact | partial | none | '
             'items all grounded | items with a "none" quote | source quoted |',
             '|' + '---|' * 12]
    for config in configs:
        cdir = os.path.join(args.runs, config)
        files = sorted(f for f in os.listdir(cdir) if f.endswith('.jsonl') and 'nodocs' not in f)
        odir = os.path.join(args.out, 'grounding', config)
        os.makedirs(odir, exist_ok=True)
        for fname in files:
            sets = {r['id']: r for r in load(os.path.join(args.sets, fname))}
            rows = []
            for r in load(os.path.join(cdir, fname)):
                item = sets[r['id']]
                user = next(m['content'] for m in item['messages'] if m['role'] == 'user')
                texts = DOC_RE.findall(user)
                if len(texts) != len(item['doc_ids']):
                    sys.exit(f'{config}/{fname} {r["id"]}: {len(texts)} <DOCUMENT> blocks, '
                             f'{len(item["doc_ids"])} doc_ids')
                docs = [norm(t) for t in texts]
                question = norm(DOC_RE.sub('', user))
                quotes = []
                for q in QUOTE_RE.findall(r['output'] or ''):
                    match, i = ground_quote(q, docs)
                    if match == 'none' and ground_quote(q, [question])[0] != 'none':
                        match = 'question'
                    quotes.append({'match': match, 'doc': item['doc_ids'][i] if i is not None else None,
                                   'chars': len(q.strip())})
                src = item.get('chunk_id')
                rows.append({'id': r['id'], 'n': len(quotes),
                             **{m: sum(q['match'] == m for q in quotes) for m in MATCHES},
                             'source_in_docs': src in item['doc_ids'] if src else None,
                             'source_quoted': any(q['doc'] == src for q in quotes) if src else None,
                             'quotes': quotes})
            with open(os.path.join(odir, fname), 'w') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            quoting = [x for x in rows if x['n']]
            tot = {m: sum(x[m] for x in rows) for m in MATCHES}
            n = sum(tot.values())
            with_src = [x for x in rows if x['source_in_docs']]
            lines.append(
                f'| {config} | {fname.removesuffix(".jsonl")} | {len(rows)} | {len(quoting)} | {n} | '
                + ' | '.join(f'{tot[m]} ({pct(tot[m], n)})' for m in MATCHES) + ' | '
                f'{sum(x["none"] == 0 for x in quoting)} ({pct(sum(x["none"] == 0 for x in quoting), len(quoting))}) | '
                f'{sum(x["none"] > 0 for x in quoting)} | '
                + (f'{sum(x["source_quoted"] for x in with_src)} of {len(with_src)}' if with_src else '—') + ' |')
    text = '\n'.join(lines) + '\n'
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'grounding_report.md'), 'w') as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--runs', required=True, help='eval_run.py output folder (holds one folder per config)')
    p.add_argument('--out', required=True)
    p.add_argument('--sets', help='eval_sets.py output folder; if given, item ids are checked against it')
    p.add_argument('--configs', nargs='+', help='config folders to read (default: all not starting with "_")')
    p = sub.add_parser('ground')
    p.add_argument('--runs', required=True, help='eval_run.py output folder (holds one folder per config)')
    p.add_argument('--sets', required=True, help='eval_sets.py output folder (prompts with the documents)')
    p.add_argument('--out', required=True)
    p.add_argument('--configs', nargs='+', help='config folders to read (default: all not starting with "_")')
    args = ap.parse_args()
    {'extract': cmd_extract, 'ground': cmd_ground}[args.cmd](args)


if __name__ == '__main__':
    main()
