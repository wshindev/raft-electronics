#!/usr/bin/env python3
"""Step 07: turn each eval item's reference answer into checkable key points.

One Claude call per eval item, once. The judge later checks model answers against these
points instead of against the free-form reference. Point kinds:
    numeric     a quantity; acceptable values with units (alternatives the reference allows)
                and a relative tolerance. The judge only extracts the model's value; Python
                compares, so numbers are graded deterministically.
    exact       a non-numeric answer that must match: hex/binary values, a formula, code, a
                choice from given options ("too high", "span shift"). Equivalent forms listed.
    conceptual  one sentence the answer must convey.
Parts of a question that can't be checked in text (sketches, "show your work") are noted,
not turned into points. A suspected error in the reference is flagged (reference_issue) and
the correct value accepted as well.

Usage:
    uv run python3 -I pipeline/keypoints.py estimate --data data/model --gen raft_heldout_fable [--limit 20]
    # pilot: synchronous calls on a fixed sample, half exam and half generated items
    uv run python3 -I pipeline/keypoints.py run --data data/model --gen raft_heldout_fable --limit 20 \\
        [--ids xq_1074 ...] --out data/model/keypoints_pilot
    # all items on the Batch API (50% price)
    uv run python3 -I pipeline/keypoints.py submit --data data/model --gen raft_heldout_fable --out data/model/keypoints
    uv run python3 -I pipeline/keypoints.py status --data data/model --gen raft_heldout_fable --out data/model/keypoints
    uv run python3 -I pipeline/keypoints.py collect --data data/model --gen raft_heldout_fable --out data/model/keypoints
    # stats, checks and review.md for the spot-check (offline)
    uv run python3 -I pipeline/keypoints.py review --data data/model --gen raft_heldout_fable --out data/model/keypoints_pilot
    # grounding check: numeric/exact points must appear in the reference; writes check.md with
    # the flagged items only (offline)
    uv run python3 -I pipeline/keypoints.py check --data data/model --gen raft_heldout_fable --out data/model/keypoints

`run`, `submit`, `status` and `collect` need ANTHROPIC_API_KEY. Results are appended to
<out>/keypoints.jsonl (latest ok record per item wins); batch IDs go to <out>/batches.jsonl.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sets import eval_sets  # noqa: E402

MODEL = 'claude-opus-5-5'
# USD per million tokens, standard rates (Batch API = 50%); claude-api skill, cached 2026-10-06
PRICES = {'claude-opus-5-5': {'in': 4.00, 'out': 20.00, 'cache_read': 0.20, 'cache_write': 5.00},
          'claude-sonnet-5-5': {'in': 2.00, 'out': 10.00, 'cache_read': 0.20, 'cache_write': 2.50}}
MAX_TOKENS = 16000
SAMPLE_SEED = 11
PROMPT_VERSION = 'kp2'  # kp2: as_written is the number only, without the unit

SYSTEM = """\
You prepare grading keys for an electronics eval. You get a question and its reference \
answer. Break the reference answer into a short list of key points that a grader can check \
one by one against a student's final answer. A student who states every point correctly \
has fully answered the question; each point should be needed for that.

Point kinds:
- numeric: a quantity the question asks for. Give every value the reference accepts in \
`values` (e.g. both results when the reference allows either resistor on top, or either \
supply voltage), each with its unit as plain text ("V", "kOhm", "mA", "W", "Hz", "dB", \
"degrees"; "" if dimensionless). Keep the reference's unit and its digits; copy the number \
as written, without its unit, into `as_written` ("12.091", "−0.04", "1.2247" for "1.2247 : 1", \
"3.8020e8" for "3.8020 × 10^8"). `rel_tol` is the relative tolerance: 0.005 by default, which \
allows ordinary intermediate rounding but not an arithmetic slip. Widen it only when the \
reference itself is approximate: rounded to one or two significant figures, "about", read \
from a graph, or dependent on a choice the question leaves open; never above 0.1. Use \
`abs_tol` (in the point's unit) only for a value of zero or one where relative tolerance \
makes no sense; otherwise 0. A ratio stated as "1.22:1" is one numeric value.
- exact: an answer that is right or wrong as a whole and is not a plain quantity: hex, \
binary or octal values, a symbolic formula, a line of code, a register value, an option \
picked from those the question offers ("too high", "span shift", "possible"). Put the \
reference's form first in `accept`, then equivalent forms a correct student might write \
(e.g. "0x6C", "6Ch", "6C hex"; "111 1101", "1111101"). Don't list forms that would accept \
a wrong answer.
- conceptual: one claim, as one plain sentence in `statement`, that the answer must \
convey. Split a two-part answer ("why ..., and what ...") into separate points. Leave out \
supporting detail, restatements of the question, and asides the question didn't ask for.

For every point, `label` is a short name ("V_in", "R2", "power factor", "why the diode"), \
and `statement` says what the point requires in words. Unused fields: empty list, 0, "".

Questions sometimes ask for things that can't be checked in a text answer (sketch the \
circuit, show your work, draw the waveform). Don't make points for those; describe them in \
`ungradable`.

Check the reference. If it contains an error (a wrong calculation, a unit conversion that \
doesn't hold, a value contradicting its own working), describe it in `reference_issue` \
with the correct value, and accept both the reference's value and the correct one so a \
correct student isn't penalized. Otherwise leave `reference_issue` empty. Don't flag \
differences in rounding or style.

Usually 1 to 4 points; more only when the question asks for more separate results.\
"""

SCHEMA = {
    'type': 'object',
    'properties': {
        'points': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'kind': {'type': 'string', 'enum': ['numeric', 'exact', 'conceptual']},
                'label': {'type': 'string'},
                'statement': {'type': 'string'},
                'values': {'type': 'array', 'items': {
                    'type': 'object',
                    'properties': {'value': {'type': 'number'}, 'unit': {'type': 'string'},
                                   'as_written': {'type': 'string'}},
                    'required': ['value', 'unit', 'as_written'], 'additionalProperties': False}},
                'rel_tol': {'type': 'number'},
                'abs_tol': {'type': 'number'},
                'accept': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['kind', 'label', 'statement', 'values', 'rel_tol', 'abs_tol', 'accept'],
            'additionalProperties': False}},
        'ungradable': {'type': 'string'},
        'reference_issue': {'type': 'string'},
    },
    'required': ['points', 'ungradable', 'reference_issue'],
    'additionalProperties': False,
}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def custom_id(item_id):
    return item_id.replace('#', '__')  # batch custom_id allows [A-Za-z0-9_-] only


def reference_text(it):
    ref = it['reference']
    if 'answer' in ref:
        return ref['answer']
    return f'{ref["final_answer"]}\n\nWorking behind it (for context):\n{ref["reasoning"]}'


def user_text(it):
    return (f'<question>\n{it["question"]}\n</question>\n\n'
            f'<reference_answer>\n{reference_text(it)}\n</reference_answer>')


def request_params(it, model, effort):
    return {
        'model': model,
        'max_tokens': MAX_TOKENS,
        'system': [{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
        'thinking': {'type': 'adaptive'},
        'output_config': {'effort': effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        'messages': [{'role': 'user', 'content': user_text(it)}],
    }


def load_items(args):
    """[(set name, item)]; with --limit, a seeded sample split evenly across the sets
    (exam: one item from each of N variant groups)."""
    sets = eval_sets(args.data, args.gen)
    out = []
    for k, (name, items) in enumerate(sets.items()):
        if args.limit:
            n = args.limit // len(sets) + (k < args.limit % len(sets))
            rng = random.Random(SAMPLE_SEED)
            if items[0].get('variant_group'):  # one item per group, so big groups don't dominate
                groups = {}
                for it in items:
                    groups.setdefault(it['variant_group'], []).append(it)
                items = [rng.choice(groups[g]) for g in rng.sample(sorted(groups), min(n, len(groups)))]
            else:
                items = rng.sample(items, min(n, len(items)))
        out += [(name, it) for it in items]
    if args.limit and args.ids:  # hand-picked cases on top of the sample
        have = {it['id'] for _, it in out}
        out += [(name, it) for name, items in sets.items() for it in items
                if it['id'] in args.ids and it['id'] not in have]
    ids =[custom_id(it['id']) for _, it in out]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', i) for i in ids)
    return out


def cost(model, inp, out, cache_read=0, cache_write=0, batch=False):
    p = PRICES[model]
    usd = (inp * p['in'] + out * p['out'] + cache_read * p['cache_read'] + cache_write * p['cache_write']) / 1e6
    return usd / 2 if batch else usd


def usage_cost(model, u, batch=False):
    return cost(model, u['input'], u['output'], u['cache_read'], u['cache_write'], batch=batch)


def cmd_estimate(args):
    items = load_items(args)
    sys_tok = len(SYSTEM + json.dumps(SCHEMA)) / 3.0
    n = len(items)
    inp = sum(len(user_text(it)) / 3.0 for _, it in items)
    print(f'{n} items ({", ".join(f"{s}: {sum(x == s for x, _ in items)}" for s in dict(items))}), '
          f'prompt {PROMPT_VERSION}')
    print(f'est. input {inp / n:,.0f} tokens/item (+ ~{sys_tok:,.0f} cached system prompt)')
    print(f'output assumed {args.out_low:,}-{args.out_high:,} tokens/item (thinking + JSON)')
    for batch in (False, True):
        lo, hi = (cost(args.model, inp, n * o, cache_read=n * sys_tok, batch=batch)
                  for o in (args.out_low, args.out_high))
        print(f'{args.model} effort {args.effort} {"batch" if batch else "standard":<9} ${lo:,.2f} - ${hi:,.2f}')


def out_path(out_dir):
    return os.path.join(out_dir, 'keypoints.jsonl')


def base_rec(set_name, it, model, effort):
    return {'id': it['id'], 'set': set_name, 'variant_group': it.get('variant_group'),
            'model': model, 'effort': effort, 'prompt_version': PROMPT_VERSION}


def parse_message(rec, msg):
    u = msg.usage
    rec.update(stop_reason=msg.stop_reason,
               usage={'input': u.input_tokens, 'output': u.output_tokens,
                      'cache_read': u.cache_read_input_tokens or 0,
                      'cache_write': u.cache_creation_input_tokens or 0})
    if msg.stop_reason != 'end_turn':
        return {**rec, 'status': 'error', 'error': f'stop_reason {msg.stop_reason}'}
    text = next((b.text for b in msg.content if b.type == 'text'), '')
    try:
        rec.update(json.loads(text))
    except json.JSONDecodeError:
        return {**rec, 'status': 'error', 'error': 'invalid JSON', 'raw': text}
    rec['status'] = 'ok'
    return rec


def gen_one(client, set_name, it, model, effort):
    import anthropic
    rec = base_rec(set_name, it, model, effort)
    try:
        with client.messages.stream(**request_params(it, model, effort)) as stream:
            msg = stream.get_final_message()
            rec['request_id'] = stream.request_id
    except anthropic.APIStatusError as e:
        return {**rec, 'status': 'error', 'error': f'{e.status_code}: {e.message}'}
    except anthropic.APIConnectionError as e:
        return {**rec, 'status': 'error', 'error': f'connection: {e}'}
    return parse_message(rec, msg)


def done_ids(out_dir):
    if not os.path.exists(out_path(out_dir)):
        return set()
    return {r['id'] for r in load(out_path(out_dir)) if r['status'] == 'ok' and r['prompt_version'] == PROMPT_VERSION}


def client_():
    import anthropic
    return anthropic.Anthropic(max_retries=4)


def cmd_run(args):
    client = client_()
    os.makedirs(args.out, exist_ok=True)
    done = done_ids(args.out)
    todo = [(s, it) for s, it in load_items(args) if it['id'] not in done]
    print(f'{len(todo)} items to do ({len(done)} done), {args.model}, effort {args.effort}, prompt {PROMPT_VERSION}')
    total = 0.0
    with open(out_path(args.out), 'a') as f, cf.ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(gen_one, client, s, it, args.model, args.effort) for s, it in todo]
        for fut in cf.as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + '\n')  # save before anything else can fail
            f.flush()
            if 'usage' in rec:
                total += usage_cost(args.model, rec['usage'])
            print(f"  {rec['id']} {rec['status']} {len(rec.get('points', []))} points"
                  + (f" ({rec['error']})" if rec['status'] != 'ok' else '') + f'  running cost ${total:.2f}')


def batches_path(out_dir):
    return os.path.join(out_dir, 'batches.jsonl')


def load_batches(out_dir):
    """Batch log: one line per submit, plus a {'batch_id', 'collected': True} line per collect."""
    batches = {}
    if os.path.exists(batches_path(out_dir)):
        for b in load(batches_path(out_dir)):
            batches.setdefault(b['batch_id'], {}).update(b)
    return batches


def log_batch(out_dir, entry):
    with open(batches_path(out_dir), 'a') as f:
        f.write(json.dumps(entry) + '\n')


def cmd_submit(args):
    client = client_()
    os.makedirs(args.out, exist_ok=True)
    done = done_ids(args.out)
    pending = {i for b in load_batches(args.out).values() if not b.get('collected') for i in b['ids']}
    todo = [(s, it) for s, it in load_items(args) if it['id'] not in done | pending]
    print(f'{len(todo)} items to submit ({len(done)} done, {len(pending)} in uncollected batches), '
          f'{args.model}, effort {args.effort}, prompt {PROMPT_VERSION}')
    if not todo:
        return
    requests = [{'custom_id': custom_id(it['id']), 'params': request_params(it, args.model, args.effort)}
                for _, it in todo]
    size = len(json.dumps(requests))
    assert len(requests) <= 100_000 and size < 200e6, (len(requests), size)  # API limits: 100k / 256 MB
    batch = client.messages.batches.create(requests=requests)
    log_batch(args.out, {'batch_id': batch.id, 'model': args.model, 'effort': args.effort,
                         'prompt_version': PROMPT_VERSION, 'ids': [it['id'] for _, it in todo]})
    print(f'submitted {batch.id}: {len(requests)} requests, {size / 1e6:.1f} MB')


def cmd_status(args):
    client = client_()
    for bid, b in load_batches(args.out).items():
        if b.get('collected'):
            print(f'{bid} collected ({len(b["ids"])} items)')
            continue
        info = client.messages.batches.retrieve(bid)
        n = info.request_counts
        print(f'{bid} {info.processing_status}: processing {n.processing}, succeeded {n.succeeded}, '
              f'errored {n.errored}, canceled {n.canceled}, expired {n.expired}')


def cmd_collect(args):
    client = client_()
    by_cid = {custom_id(it['id']): (s, it) for s, it in load_items(args)}
    for bid, b in load_batches(args.out).items():
        if b.get('collected'):
            continue
        status = client.messages.batches.retrieve(bid).processing_status
        if status != 'ended':
            print(f'{bid} {status}, not collected')
            continue
        counts, total = {}, 0.0
        with open(out_path(args.out), 'a') as f:
            for res in client.messages.batches.results(bid):
                rec = base_rec(*by_cid[res.custom_id], b['model'], b['effort'])
                rec.update(batch_id=bid, batch=True, prompt_version=b['prompt_version'])
                t = res.result.type
                if t == 'succeeded':
                    rec = parse_message(rec, res.result.message)
                    total += usage_cost(b['model'], rec['usage'], batch=True)
                elif t == 'errored':
                    e = res.result.error.error
                    rec.update(status='error', error=f'{e.type}: {e.message}')
                else:  # canceled / expired: not billed, resubmit
                    rec.update(status='error', error=t)
                f.write(json.dumps(rec) + '\n')
                counts[rec['status']] = counts.get(rec['status'], 0) + 1
        log_batch(args.out, {'batch_id': bid, 'collected': True})
        print(f'{bid} collected: {counts}, cost ${total:.2f}')


NUM = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+'
SCI = re.compile(rf'({NUM})\s*(?:\\times|×|x|\*)\s*10\s*\^\s*\{{?\s*([-+]?\d+)\s*\}}?')
POW = re.compile(r'(\d+(?:\.\d+)?)\s*\^\s*\{?\s*([-+]?\d+(?:\.\d+)?)\s*\}?')
WORDS = {w: k for k, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split())}
SCALE = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9}


def to_float(s):
    return float(s.replace(',', ''))


def parse_number(s):
    """as_written -> float ("−0.04", "380,204,032", "3.8020 × 10^8", "3.8e8", "one"); ValueError if not a number."""
    s = s.strip().replace('−', '-').replace('$', '')
    if s.lower() in WORDS:
        return float(WORDS[s.lower()])
    m = re.fullmatch(rf'({NUM})\s*({"|".join(SCALE)})', s.lower())
    if m:
        return to_float(m[1]) * SCALE[m[2]]
    m = SCI.fullmatch(s.lstrip('+-'))
    if m:
        return (-1 if s.startswith('-') else 1) * to_float(m[1]) * 10 ** int(m[2])
    return to_float(s)


def reference_numbers(text):
    """Absolute values of every number in a reference text: plain numbers, a × 10^n, a^b, number words."""
    t = text.replace('−', '-').replace('$', '').replace('**', '')
    nums = {to_float(m) for m in re.findall(NUM, t)}
    nums |= {to_float(a) * 10 ** int(b) for a, b in SCI.findall(t)}
    nums |= {float(a) ** float(b) for a, b in POW.findall(t) if float(b) * len(a) < 300}
    nums |= {float(re.sub(r'[^\d.eE+-]', '', m)) for m in re.findall(r'\d(?:\.\d+)?[eE][-+]?\d+', t)}
    nums |= {float(WORDS[w]) for w in re.findall(r'[a-z]+', t.lower()) if w in WORDS}
    nums |= {to_float(a) * SCALE[b] for a, b in re.findall(rf'({NUM})\s*({"|".join(SCALE)})', t.lower())}
    return nums


def norm_exact(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def point_problems(p, version=PROMPT_VERSION):
    """Format problems in one point (the schema can't express these)."""
    probs = []
    if p['kind'] == 'numeric':
        if not p['values']:
            probs.append('numeric without values')
        exact_int = p['rel_tol'] == 0 and all(float(v['value']).is_integer() for v in p['values'])  # counts, addresses
        if not (0 < p['rel_tol'] <= 0.1) and not p['abs_tol'] > 0 and not exact_int:
            probs.append(f'rel_tol {p["rel_tol"]}')
        for v in p['values']:
            try:
                if abs(parse_number(v['as_written']) - v['value']) > 1e-9 * max(1, abs(v['value'])):
                    probs.append(f'value {v["value"]} != as_written {v["as_written"]!r}')
            except ValueError:
                if version != 'kp1':  # kp1 wrote the unit into as_written ("3.99 V")
                    probs.append(f'as_written not a number: {v["as_written"]!r}')
    elif p['kind'] == 'exact' and not p['accept']:
        probs.append('exact without accept forms')
    if p['kind'] != 'numeric' and p['values']:
        probs.append(f'{p["kind"]} with values')
    return probs


def fmt_point(p):
    if p['kind'] == 'numeric':
        vals = ' | '.join(f'{v["as_written"]} {v["unit"]}'.strip() for v in p['values'])
        tol = f'±{p["rel_tol"]:.1%}' + (f', abs ±{p["abs_tol"]:g}' if p['abs_tol'] else '')
        return f'**numeric** `{p["label"]}` = {vals} ({tol}): {p["statement"]}'
    if p['kind'] == 'exact':
        return f'**exact** `{p["label"]}` ∈ {p["accept"]}: {p["statement"]}'
    return f'**conceptual** `{p["label"]}`: {p["statement"]}'


def cmd_review(args):
    if not os.path.exists(out_path(args.out)):
        sys.exit(f'{out_path(args.out)} missing')
    items = {it['id']: (s, it) for s, it in load_items(args)}
    latest = {}
    for r in load(out_path(args.out)):
        if r['status'] == 'ok' or r['id'] not in latest or latest[r['id']]['status'] != 'ok':
            latest[r['id']] = r
    recs = [r for r in latest.values() if r['id'] in items]
    ok = [r for r in recs if r['status'] == 'ok']
    spent = sum(usage_cost(r['model'], r['usage'], r.get('batch', False)) for r in recs if 'usage' in r)
    print(f'{len(ok)} ok of {len(recs)} records; {len(items) - len(recs)} items without a record; spent ${spent:.2f}')
    if ok:
        n = len(ok)
        out_tok = sorted(r['usage']['output'] for r in ok)
        print(f'output tokens/item: mean {sum(out_tok) / n:,.0f}, median {out_tok[n // 2]:,}, max {out_tok[-1]:,}')
        n_all = len(load_items(argparse.Namespace(**{**vars(args), 'limit': 0})))
        per = spent / len(recs) * (2 if ok[0].get('batch') else 1)  # standard rate per item
        print(f'cost/item ${per:.4f} standard -> all {n_all} items: standard ~${per * n_all:,.2f}, '
              f'batch ~${per * n_all / 2:,.2f}')
    lines = [f'# Key points review ({args.out})', '']
    stats = {}
    for s in sorted({r['set'] for r in ok}):
        rs = [r for r in ok if r['set'] == s]
        pts = [p for r in rs for p in r['points']]
        kinds = {k: sum(p['kind'] == k for p in pts) for k in ('numeric', 'exact', 'conceptual')}
        stats[s] = (f'{s}: {len(rs)} items, {len(pts)} points ({len(pts) / len(rs):.1f}/item; {kinds}), '
                    f'numeric with alternatives {sum(len(p["values"]) > 1 for p in pts)}, '
                    f'ungradable noted {sum(bool(r["ungradable"]) for r in rs)}, '
                    f'reference issues {sum(bool(r["reference_issue"]) for r in rs)}')
        print(stats[s])
        lines.append(f'- {stats[s]}')
    problems = [(r['id'], pr) for r in ok for p in r['points'] for pr in point_problems(p, r['prompt_version'])]
    print(f'format problems: {len(problems)}' + ''.join(f'\n  {i}: {pr}' for i, pr in problems))
    lines += [f'- format problems: {len(problems)}', '']
    for r in sorted(ok, key=lambda r: (r['set'], r['id'])):
        s, it = items[r['id']]
        lines += [f'## {r["id"]} ({s}' + (f', {r["variant_group"]}' if r.get('variant_group') else '') + ')', '',
                  '**Question:** ' + it['question'], '',
                  '**Reference:** ' + (it['reference'].get('answer') or it['reference']['final_answer']), '',
                  '**Key points:**', *[f'{k}. {fmt_point(p)}' for k, p in enumerate(r['points'], 1)], '']
        if r['ungradable']:
            lines += [f'**Ungradable:** {r["ungradable"]}', '']
        if r['reference_issue']:
            lines += [f'**Reference issue:** {r["reference_issue"]}', '']
        lines.append('---')
    for r in recs:
        if r['status'] != 'ok':
            print(f'  {r["id"]}: {r["error"]}')
    path = os.path.join(args.out, 'review.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def grounding_flags(r, it):
    """Reasons a record needs a human look: numeric values or exact forms not found in the
    reference, format problems, a reference issue, or no points at all. Conceptual points
    can't be checked this way."""
    ref = reference_text(it)
    nums, ref_norm = reference_numbers(ref), norm_exact(ref)
    flags = [f'format: {pr}' for p in r['points'] for pr in point_problems(p, r['prompt_version'])]
    for p in r['points']:
        if p['kind'] == 'numeric':
            for v in p['values']:
                tol = max(p['rel_tol'] * abs(v['value']), p['abs_tol'], 1e-9)
                if not any(abs(n - abs(v['value'])) <= tol for n in nums):
                    flags.append(f'numeric `{p["label"]}` = {v["as_written"]} {v["unit"]}: value not in the reference')
        elif p['kind'] == 'exact' and not any(norm_exact(a) and norm_exact(a) in ref_norm for a in p['accept']):
            flags.append(f'exact `{p["label"]}`: none of {p["accept"]} in the reference')
    if not r['points']:
        flags.append('no points' + (f' (ungradable: {r["ungradable"]})' if r['ungradable'] else ''))
    if r['reference_issue']:
        flags.append(f'reference issue: {r["reference_issue"]}')
    return flags


def cmd_check(args):
    if not os.path.exists(out_path(args.out)):
        sys.exit(f'{out_path(args.out)} missing')
    items = {it['id']: (s, it) for s, it in load_items(args)}
    latest = {}
    for r in load(out_path(args.out)):
        if r['status'] == 'ok' and r['id'] in items:
            latest[r['id']] = r
    missing = sorted(set(items) - set(latest))
    flagged = [(r, f) for r in sorted(latest.values(), key=lambda r: (r['set'], r['id']))
               if (f := grounding_flags(r, items[r['id']][1]))]
    pts = [p for r in latest.values() for p in r['points']]
    kinds = {k: sum(p['kind'] == k for p in pts) for k in ('numeric', 'exact', 'conceptual')}
    summary = (f'{len(latest)} items with ok key points ({len(missing)} without), {len(pts)} points {kinds}; '
               f'{len(flagged)} items flagged')
    print(summary)
    lines = [f'# Key points check ({args.out})', '', summary, '',
             'Numeric values and exact forms are looked up in the reference text (numbers within the '
             "point's tolerance, sign ignored; exact forms as substrings, letters and digits only). "
             'Conceptual points are not checked here.', '']
    if missing:
        lines += [f'Without ok key points: {", ".join(missing)}', '']
    for r, flags in flagged:
        s, it = items[r['id']]
        q = ' '.join(it['question'].split())
        lines += [f'## {r["id"]} ({s})', '',
                  f'**Q:** {q[:300]}' + ('…' if len(q) > 300 else ''), '',
                  '**Reference:** ' + ' '.join((it['reference'].get('answer') or it['reference']['final_answer']).split()), '',
                  *[f'- {fmt_point(p)}' for p in r['points']], '',
                  *[f'> ⚑ {f}' for f in flags], '', '---']
    path = os.path.join(args.out, 'check.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('estimate', 'run', 'submit', 'status', 'collect', 'review', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='data/<source> folder')
        p.add_argument('--gen', required=True, help='generate.py output dir of the held-out questions, relative to --data')
        p.add_argument('--limit', type=int, default=0, help='seeded sample of N items, split evenly across the sets')
        p.add_argument('--ids', nargs='+', default=[], help='with --limit: item ids added to the sample')
        p.add_argument('--model', default=MODEL, choices=sorted(PRICES))
        p.add_argument('--effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
        if name != 'estimate':
            p.add_argument('--out', required=True)
        if name == 'estimate':
            p.add_argument('--out-low', type=int, default=600, help='assumed output tokens per item, low')
            p.add_argument('--out-high', type=int, default=2000, help='assumed output tokens per item, high')
        if name == 'run':
            p.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    {'estimate': cmd_estimate, 'run': cmd_run, 'submit': cmd_submit, 'status': cmd_status,
     'collect': cmd_collect, 'review': cmd_review, 'check': cmd_check}[args.cmd](args)


if __name__ == '__main__':
    main()
