#!/usr/bin/env python3
"""Step 07: training answers written by a local model. Stdlib only.

The first run's training answers (Claude, ~120-140 tokens of written explanation) were too
brief for E4B (Test A). Here a local model served by llama-server writes new answers to the
same training examples: same question, same documents, same system prompt (assemble.py's
records, unchanged), thinking off, greedy. The visible answer is the training text. With
Gemma 4 31B served this is the teacher pilot; with E4B served it is self-distillation.

  run       answer each example of <sft>/<split>.jsonl (seeded --sample, or all). Resumes;
            a folder can't mix models or settings (meta.json).
  check     a judge model (31B) compares each answer's final answer with the reference final
            answer of the same record (the Claude answer), JSON schema: match / partial /
            mismatch / unclear. Can run on another server than `run`.
  assemble  keep an answer if it finished (finish_reason stop), has an "<ANSWER>" marker, the
            check says match, every quote is found verbatim in the record's documents
            (score.py ground_quote, normalised; surrounding quotation marks ignored) and, when the oracle is present, it quotes at
            least once. Writes <to>/train.jsonl + dev.jsonl in assemble.py's format with the
            assistant turn replaced (train.py reads them).
  report    counts per filter, acceptance by answer type x oracle, answer length vs the
            reference answers, seconds per item (for sizing a full run). Offline.

Usage:
    uv run python3 -I pipeline/teacher.py run --url http://HOST:8080 --sft data/model/sft_sonnet_hybrid \\
        --split train --sample 100 --out data/model/teacher/gemma31b [--workers 1]
    uv run python3 -I pipeline/teacher.py check --url http://HOST:8080 --sft data/model/sft_sonnet_hybrid \\
        --out data/model/teacher/gemma31b
    uv run python3 -I pipeline/teacher.py assemble --sft data/model/sft_sonnet_hybrid \\
        --out data/model/teacher/gemma31b --to data/model/sft_gemma31b_pilot
    uv run python3 -I pipeline/teacher.py report --sft data/model/sft_sonnet_hybrid --out data/model/teacher/gemma31b

Writes to --out: meta.json, answers_<split>.jsonl {id, split, output, reasoning, finish_reason,
prompt_tokens, completion_tokens, seconds}, checks_<split>.jsonl {id, split, model, verdict,
note, answer, method, ...}, report.md. LLAMA_API_KEY if the server needs it.
"""
import argparse
import collections
import json
import os
import random
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_run import request, server_info  # noqa: E402
from score import DOC_RE, QUOTE_RE, extract, ground_quote, norm  # noqa: E402

SEED = 20261108
SPLITS = ('train', 'dev')
CHECK_VERSION = 'c1'
CHECK_MAX_TOKENS = 512
CHECK_SYSTEM = (
    'You compare a candidate\'s final answer with the reference answer to the same question. '
    'verdict "match": the candidate states the same result as the reference (numbers equal up to rounding, '
    'about 1%; units may be written differently or converted; the same conclusion for a conceptual question; '
    'extra correct detail is fine). "partial": it gets part of the reference\'s result and misses or '
    'contradicts another part. "mismatch": it contradicts the reference or answers a different question. '
    '"unclear": the reference itself looks wrong or ambiguous. Judge only the final answers, not style. '
    'note: one short sentence.')
CHECK_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['verdict', 'note'],
                'properties': {'verdict': {'type': 'string', 'enum': ['match', 'partial', 'mismatch', 'unclear']},
                               'note': {'type': 'string'}}}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def records(sft, split):
    return load(os.path.join(sft, f'{split}.jsonl'))


def ref_answer(rec):
    return rec['messages'][-1]['content'].split('<ANSWER>:', 1)[-1].strip()


def question(rec):
    return rec['messages'][1]['content'].rsplit('\n\nQuestion: ', 1)[-1].strip()


def latest(path, ok=lambda r: True):
    """Latest record per id (a resumed run appends)."""
    out = {}
    if os.path.exists(path):
        for r in load(path):
            if ok(r):
                out[r['id']] = r
    return out


def answers_path(out, split):
    return os.path.join(out, f'answers_{split}.jsonl')


def checks_path(out, split):
    return os.path.join(out, f'checks_{split}.jsonl')


def one_model(url, timeout):
    info = server_info(url, timeout)
    if not isinstance(info['models'], list) or len(info['models']) != 1:
        sys.exit(f'server not reachable or not one model: {info["models"]}\n(--url needs http://; a 401 means '
                 f'LLAMA_API_KEY is missing or wrong)')
    return info


def run_pool(todo, call, path, workers, label):
    """Run call() over todo, appending each result to path; stop at the first server error."""
    with open(path, 'a') as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(call, it): it for it in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except (RuntimeError, urllib.error.URLError, TimeoutError) as e:
                for other in futs:
                    other.cancel()
                sys.exit(f'{futs[fut]["id"]}: {e}\nRerun the same command to resume.')
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(f'\r  {label} {n}/{len(todo)}', end='', flush=True)
    print(f'\n  wrote {path}')


# ---------- run ----------

def gen_payload(args, rec):
    return {'messages': rec['messages'][:-1], 'temperature': 0.0, 'top_k': 1, 'seed': 0,
            'max_tokens': args.max_tokens, 'cache_prompt': True,
            'chat_template_kwargs': {'enable_thinking': args.think}}


def answer(args, split, rec):
    t0 = time.time()
    res = request(args.url, '/v1/chat/completions', gen_payload(args, rec), args.timeout)
    ch, u = res['choices'][0], res.get('usage', {})
    return {'id': rec['id'], 'split': split, 'output': (ch['message'].get('content') or '').strip(),
            'reasoning': ch['message'].get('reasoning_content'), 'finish_reason': ch.get('finish_reason'),
            'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens'),
            'seconds': round(time.time() - t0, 2)}


def cmd_run(args):
    os.makedirs(args.out, exist_ok=True)
    info = one_model(args.url, args.timeout)
    meta_path = os.path.join(args.out, 'meta.json')
    params = {k: v for k, v in gen_payload(args, {'messages': [None]}).items() if k != 'messages'}
    new = {'model': info['models'][0], 'sft': os.path.normpath(args.sft), 'params': params}
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    diff = [f'  {k}: earlier {meta[k]} | now {v}' for k, v in new.items() if k in meta and meta[k] != v]
    if diff:
        sys.exit(f'{meta_path}: differs from the earlier run in this folder:\n' + '\n'.join(diff) +
                 '\nUse another --out.')
    meta.update(new, server=info)
    rows = records(args.sft, args.split)
    random.Random(f'{SEED}-{args.split}').shuffle(rows)  # a larger --sample extends a smaller one
    rows = rows[:args.sample or None]
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    path = answers_path(args.out, args.split)
    done = latest(path)
    todo = [r for r in rows if r['id'] not in done]
    print(f'{args.split}: {len(rows) - len(todo)} done, {len(todo)} to go; {new["model"]}, '
          f'thinking {"on" if args.think else "off"}')
    run_pool(todo, lambda r: answer(args, args.split, r), path, args.workers, args.split)
    cut = sum(r['finish_reason'] == 'length' for r in latest(path).values())
    if cut:
        print(f'  {cut} answers hit max tokens ({args.max_tokens}); assemble drops them')


# ---------- check ----------

def check_user(rec, ans):
    return (f'Question:\n{question(rec)}\n\nReference final answer:\n{ref_answer(rec)}\n\n'
            f'Candidate final answer:\n{ans}')


def check_one(args, model, split, rec, a):
    ans, method, _, _ = extract(a['output'])
    out = {'id': rec['id'], 'split': split, 'model': model, 'check_version': CHECK_VERSION,
           'method': method, 'answer': ans}
    payload = {'messages': [{'role': 'system', 'content': CHECK_SYSTEM},
                            {'role': 'user', 'content': check_user(rec, ans)}],
               'temperature': 0.0, 'top_k': 1, 'seed': 0, 'max_tokens': CHECK_MAX_TOKENS, 'cache_prompt': True,
               'chat_template_kwargs': {'enable_thinking': False},
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'check', 'schema': CHECK_SCHEMA}}}
    res = request(args.url, '/v1/chat/completions', payload, args.timeout)
    ch = res['choices'][0]
    try:
        v = json.loads(ch['message'].get('content') or '')
        return {**out, 'status': 'ok', 'verdict': v['verdict'], 'note': v['note']}
    except (ValueError, KeyError, TypeError):
        return {**out, 'status': 'error', 'error': f'unparsable ({ch.get("finish_reason")}): '
                                                   f'{(ch["message"].get("content") or "")[:200]!r}'}


def cmd_check(args):
    model = one_model(args.url, args.timeout)['models'][0]
    for split in SPLITS:
        ans = latest(answers_path(args.out, split))
        if not ans:
            continue
        path = checks_path(args.out, split)
        done = latest(path, lambda r: r['status'] == 'ok')
        if any(r['model'] != model or r['check_version'] != CHECK_VERSION for r in done.values()):
            sys.exit(f'{path} holds checks by another model or prompt version; move it aside first')
        recs = {r['id']: r for r in records(args.sft, split)}
        todo = [{'id': i} for i in ans if i not in done]
        print(f'{split}: {len(done)} checked, {len(todo)} to go; checker {model}, prompt {CHECK_VERSION}')
        run_pool(todo, lambda t: check_one(args, model, split, recs[t['id']], ans[t['id']]), path, args.workers,
                 split)
        bad = [r['id'] for r in latest(path).values() if r['status'] != 'ok']
        if bad:
            print(f'  {len(bad)} unparsable checks (rerun to retry): {bad[:5]}')


# ---------- assemble / report ----------

def judge(rec, a, c):
    """(reason, n_quotes): reason is 'ok' or why the answer is dropped (first failing filter)."""
    if a['finish_reason'] != 'stop':
        return 'cut_off', 0
    _, method, _, _ = extract(a['output'])
    if method not in ('colon', 'no_colon'):
        return 'no_answer_marker', 0
    if c is None or c['status'] != 'ok':
        return 'not_checked', 0
    if c['verdict'] != 'match':
        return f'check_{c["verdict"]}', 0
    docs = [norm(d) for d in DOC_RE.findall(rec['messages'][1]['content'])]
    quotes = QUOTE_RE.findall(a['output'])
    # 31B often wraps the quote in quotation marks inside the tags; those aren't part of the source text
    if any(ground_quote(q.strip().strip('"\u201c\u201d\'').strip(), docs)[0] != 'exact' for q in quotes):
        return 'quote_not_in_docs', len(quotes)
    if rec['oracle_present'] and not quotes:
        return 'no_quote', 0
    return 'ok', len(quotes)


def evaluate(args):
    """Per split: list of (record, answer, check, reason, n_quotes) for every answered record."""
    out = {}
    for split in SPLITS:
        ans = latest(answers_path(args.out, split))
        if not ans:
            continue
        checks = latest(checks_path(args.out, split))
        rows = []
        for rec in records(args.sft, split):
            if rec['id'] in ans:
                a, c = ans[rec['id']], checks.get(rec['id'])
                rows.append((rec, a, c, *judge(rec, a, c)))
        out[split] = rows
    if not out:
        sys.exit(f'no answers in {args.out}')
    return out


def cmd_assemble(args):
    meta = json.load(open(os.path.join(args.out, 'meta.json')))
    os.makedirs(args.to, exist_ok=True)
    for split, rows in evaluate(args).items():
        kept = []
        for rec, a, _, reason, _ in rows:
            if reason == 'ok':
                msgs = rec['messages'][:-1] + [{'role': 'assistant', 'content': a['output']}]
                kept.append({**rec, 'generator': meta['model'], 'messages': msgs})
        with open(os.path.join(args.to, f'{split}.jsonl'), 'w') as f:
            for r in kept:
                f.write(json.dumps(r) + '\n')
        print(f'{split}: kept {len(kept)} of {len(rows)} answered')
    with open(os.path.join(args.to, 'teacher_source.json'), 'w') as f:
        json.dump({'teacher_dir': os.path.normpath(args.out), 'sft': os.path.normpath(args.sft), 'meta': meta,
                   'check_version': CHECK_VERSION}, f, indent=1)
    print(f'wrote {args.to}')


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def cmd_report(args):
    meta = json.load(open(os.path.join(args.out, 'meta.json')))
    lines = [f'# Teacher answers: {meta["model"]}', '',
             f'sft {meta["sft"]}, params {json.dumps(meta["params"])}, check prompt {CHECK_VERSION}', '']
    for split, rows in evaluate(args).items():
        n = len(rows)
        reasons = collections.Counter(r[3] for r in rows)
        lines += [f'## {split}: {n} answered, {reasons["ok"]} kept ({reasons["ok"] / n:.0%})', '',
                  '| filter (first failing) | n |', '|---|---|']
        lines += [f'| {k} | {v} |' for k, v in reasons.most_common()]
        verdicts = collections.Counter(c['verdict'] for _, _, c, _, _ in rows if c and c['status'] == 'ok')
        lines += ['', f'Check verdicts (all checked answers): {dict(verdicts.most_common())}', '',
                  '| answer type, oracle | n | kept |', '|---|---|---|']
        groups = collections.defaultdict(list)
        for rec, _, _, reason, _ in rows:
            groups[(rec['answer_type'], rec['oracle_present'])].append(reason == 'ok')
        lines += [f'| {t}, {"present" if o else "absent"} | {len(v)} | {sum(v)} ({sum(v) / len(v):.0%}) |'
                  for (t, o), v in sorted(groups.items())]
        tok = [a['completion_tokens'] or 0 for _, a, _, _, _ in rows]
        kept_tok = [a['completion_tokens'] or 0 for _, a, _, r, _ in rows if r == 'ok']
        ch = [len(a['output']) for _, a, _, _, _ in rows]
        ref_ch = [len(rec['messages'][-1]['content']) for rec, _, _, _, _ in rows]
        quotes = [q for *_, r, q in rows if r == 'ok']
        sec = [a['seconds'] for _, a, _, _, _ in rows]
        lines += ['', f'Answer length: median {median(tok)} tokens (kept {median(kept_tok)}), max {max(tok)}; '
                      f'chars median {median(ch)} vs reference answers {median(ref_ch)} '
                      f'({median(ch) / max(median(ref_ch), 1):.1f}x).',
                  f'Quotes per kept answer: median {median(quotes)}.',
                  f'Seconds per answer: median {median(sec):.1f}, mean {sum(sec) / n:.1f} '
                  f'(at --workers > 1 these overlap).', '']
        drop = [(rec, a, c, r) for rec, a, c, r, _ in rows if r.startswith('check_') and r != 'check_match']
        if drop:
            lines += ['Examples of check failures (first 10):', '']
            for rec, a, c, r in drop[:10]:
                lines += [f'- `{rec["id"]}` {r}: {c["note"]}  ',
                          f'  reference: {ref_answer(rec)[:200]}  ',
                          f'  candidate: {c["answer"][:200]}']
            lines.append('')
    path = os.path.join(args.out, 'report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'check', 'assemble', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--sft', required=True, help='assemble.py output folder with the reference answers')
        p.add_argument('--out', required=True, help='teacher folder, e.g. data/model/teacher/gemma31b')
        if name in ('run', 'check'):
            p.add_argument('--url', required=True, help='llama-server base URL (start it with --jinja)')
            p.add_argument('--timeout', type=float, default=900)
            p.add_argument('--workers', type=int, default=1, help='concurrent requests (match the server -np)')
        if name == 'run':
            p.add_argument('--split', choices=SPLITS, default='train')
            p.add_argument('--sample', type=int, default=0, help='seeded sample of N records (0 = all)')
            p.add_argument('--max-tokens', type=int, default=4096)
            p.add_argument('--think', action='store_true', help='thinking on (default off: the visible answer is '
                                                                'the training text)')
        if name == 'assemble':
            p.add_argument('--to', required=True, help='output SFT folder, e.g. data/model/sft_gemma31b_pilot')
    args = ap.parse_args()
    {'run': cmd_run, 'check': cmd_check, 'assemble': cmd_assemble, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
