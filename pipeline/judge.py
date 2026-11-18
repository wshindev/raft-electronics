#!/usr/bin/env python3
"""Step 07: judge model answers against the key points.

One Claude call per (config, eval file, item). The judge sees the question, the key points and
the model's final answer (from `score.py extract`). Per key point it returns a quote from the
answer and a verdict:
    numeric     the expected value is masked; the judge only reports the model's value (in a
                unit the point lists). Python compares it with the key values and tolerance.
                Where the point has the default tolerance (<= 0.5%), the value must also round
                to the key at the precision written: |answer - key| <= half a unit in the last
                digit of the answer (read from the judge's quote) plus half a unit in the last
                digit of the key's as_written (both are roundings of the true value).
                So 0.662 or 0.66 pass for 0.661619, 0.664 fails; 10367.2 passes for 10367.3.
    exact       covered / wrong / missing
    conceptual  covered / partly / wrong / missing
Scoring (here, not in the judge): covered 1, partly 0.5, else 0. Item score = mean over points.

Usage:
    # validation sample: same seeded, stratified sample for every judge and for the hand grades
    uv run python3 -I pipeline/judge.py estimate --data data/model --gen raft_heldout_fable \\
        --scores data/model/scores --keypoints data/model/keypoints [--sample 204] [--judge haiku]
    uv run python3 -I pipeline/judge.py run ... --sample 204 --judge haiku --out data/model/scores/judge_val/haiku
    uv run python3 -I pipeline/judge.py run ... --sample 204 --judge sonnet --out data/model/scores/judge_val/sonnet
    # hand grades on the first N items of the same sample (interactive, offline)
    uv run python3 -I pipeline/judge.py grade ... --sample 204 --n 20 --out data/model/scores/judge_val/human.jsonl
    # agreement between two judges, or a judge and the hand grades (offline)
    uv run python3 -I pipeline/judge.py agree ... --sample 204 --a data/model/scores/judge_val/haiku \\
        --b data/model/scores/judge_val/sonnet [--human data/model/scores/judge_val/human.jsonl]
    # full grid on the Batch API
    uv run python3 -I pipeline/judge.py submit ... --judge haiku --out data/model/scores/judge/haiku
    uv run python3 -I pipeline/judge.py {status|collect} ... --judge haiku --out data/model/scores/judge/haiku
    # score table per config x file, paired top5 vs nodocs, with quote grounding (offline)
    uv run python3 -I pipeline/judge.py report ... --out data/model/scores/judge/haiku [--grounding data/model/scores/grounding]

Local judge (no API): `run --judge local --url http://HOST:8080` sends the same system prompt,
user text and JSON schema to a llama-server (/v1/chat/completions, greedy, `response_format`
json_schema; start the server with --jinja). Thinking is off unless --think. The server's model
name goes into each record; LLAMA_API_KEY if the server needs it. Validate it like the Claude
judges: `run ... --sample 204 --judge local --url ... --out data/model/scores/judge_val/<tag>`,
then `agree --a <that folder> --b .../haiku --human .../human.jsonl`. Pass
`--configs e4b_base e4b_opus e4b_sonnet` with --sample 204: later configs (e4b_base_brief)
change the seeded sample, and only those three reproduce the validated one.

`run`, `submit`, `status` and `collect` need ANTHROPIC_API_KEY (except `run --judge local`). Judgments are appended to
<out>/judgments.jsonl (latest ok record per key wins); batch IDs go to <out>/batches.jsonl.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sets import eval_sets  # noqa: E402
from eval_run import request, server_info  # noqa: E402
from keypoints import NUM, SCALE, SCI, WORDS, load, reference_numbers, to_float  # noqa: E402

JUDGES = {'haiku': 'claude-haiku-5-5', 'sonnet': 'claude-sonnet-5-5'}
# USD per million tokens, standard rates (Batch API = 50%); claude-api skill, cached 2026-10-06
PRICES = {'claude-haiku-5-5': {'in': 0.10, 'out': 0.50, 'cache_read': 0.01, 'cache_write': 0.125},
          'claude-sonnet-5-5': {'in': 2.00, 'out': 10.00, 'cache_read': 0.20, 'cache_write': 2.50}}
MAX_TOKENS = 16000
SAMPLE_SEED = 7
PROMPT_VERSION = 'j1'
FILE_SET = {'exam': 'eval_exam'}  # answers' "set" field -> eval_sets name; anything else is the --gen set
VERDICTS = ['stated', 'conflicting', 'covered', 'partly', 'wrong', 'missing']
MASK = '[value]'

SYSTEM = """\
You grade a student's answer to an electronics question against a fixed list of key points. \
You see the question, the key points and the student's final answer. Grade only what the \
answer states. Don't solve the question yourself, and don't give credit for what the student \
might have meant.

For each key point, in order, return:
- `quote`: the shortest passage of the answer that addresses this point, copied exactly; "" \
if the answer doesn't address it.
- `verdict` and `values`, depending on the point's kind:
  - numeric: the expected value is hidden from you (shown as [value]); report what the student \
gave. `values`: the student's final value for this quantity as {unit, value}, with `unit` one \
of the units listed for the point (prefer the one the student used; convert prefixes if \
needed, e.g. 2.67 mA -> 0.00267 with unit "A"). Keep the sign the student gives, including a \
sign stated in words ("0.04 % low" -> -0.04; "lagging by 90 degrees" -> -90). If the point \
lists a magnitude unit and "degrees" (polar form), give two entries: magnitude and angle. If \
the student gives the quantity only in a form that fixes it directly (e.g. a turns pair 40 \
and 49 when the point is the ratio Ns/Np), compute it. `verdict`: "stated" if the answer gives \
a value, "conflicting" if it gives two or more different final values for the quantity, \
"missing" if it gives none (then `values` is empty). Never fill in a value the answer \
doesn't give.
  - exact: "covered" if the answer gives the required result in one of the accepted forms or \
an equivalent one (same number in another notation, the same formula rearranged, the same \
choice in other words); "wrong" if it gives a different result; "missing" if it gives none.
  - conceptual: "covered" if the answer conveys the claim, in any wording; "partly" if it \
conveys only part of it, or too vaguely to tell; "wrong" if it contradicts it; "missing" if \
it doesn't address it.
  For exact and conceptual points, `values` is empty.
- `note`: one short sentence on why, for any verdict other than "covered" or "stated"; \
otherwise "".

Material beyond the key points (working, other quantities, asides) is ignored unless it \
contradicts a point. The answer may be cut off, or may be the end of a longer response with \
no marked final answer; grade what is there.\
"""

SCHEMA = {
    'type': 'object',
    'properties': {'points': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {
            'point': {'type': 'integer'},
            'quote': {'type': 'string'},
            'verdict': {'type': 'string', 'enum': VERDICTS},
            'values': {'type': 'array', 'items': {
                'type': 'object',
                'properties': {'unit': {'type': 'string'}, 'value': {'type': 'number'}},
                'required': ['unit', 'value'], 'additionalProperties': False}},
            'note': {'type': 'string'},
        },
        'required': ['point', 'quote', 'verdict', 'values', 'note'], 'additionalProperties': False}}},
    'required': ['points'],
    'additionalProperties': False,
}


# ---------- inputs ----------

def unit_label(u):
    return u or '(no unit)'


def mask_values(statement, values):
    """Replace the given key values (all numeric points of the item) in a statement with [value]."""
    vals = [abs(v['value']) for v in values]

    def hit(x, decimals=None):  # statements often round the value ("about 4.898" for 4.8977)
        return any(math.isclose(x, v, rel_tol=0.01, abs_tol=1e-12)
                   or (decimals is not None and round(v, decimals) == x) for v in vals)

    def num(m):
        t = m[0]
        return MASK if hit(to_float(t), len(t.split('.')[1]) if '.' in t else 0) else t
    s = SCI.sub(lambda m: MASK if hit(to_float(m[1]) * 10 ** int(m[2])) else m[0], statement)  # "3.8020 × 10^8"
    s = re.sub(rf'({NUM})\s*({"|".join(SCALE)})\b', lambda m: MASK if hit(to_float(m[1]) * SCALE[m[2]]) else m[0], s)
    s = re.sub(rf'\b({"|".join(WORDS)})\b', lambda m: MASK if hit(float(WORDS[m[0].lower()])) else m[0], s, flags=re.I)
    return re.sub(rf'(?:(?<![\w.])|(?<=\bj))(?:{NUM})(?!\w|\.\d)', num, s)  # "j1.4765": imaginary part


def fmt_points(points):
    """Numeric values are masked in numeric and conceptual statements (a statement can mention another
    point's value); exact points keep their accepted forms, which the judge needs."""
    values = [v for p in points if p['kind'] == 'numeric' for v in p['values']]
    lines = []
    for k, p in enumerate(points, 1):
        if p['kind'] == 'numeric':
            units = ' or '.join(dict.fromkeys(unit_label(v['unit']) for v in p['values']))
            lines.append(f'{k}. [numeric] {p["label"]} (unit: {units}): {mask_values(p["statement"], values)}')
        elif p['kind'] == 'exact':
            lines.append(f'{k}. [exact] {p["label"]}: {p["statement"]} Accepted forms: {"; ".join(p["accept"])}')
        else:
            lines.append(f'{k}. [conceptual] {p["label"]}: {mask_values(p["statement"], values)}')
    return '\n'.join(lines)


NOTES = {'tail': 'The response had no marked final answer; this is the end of the response.',
         'before_marker': 'The response wrote its answer before an empty final-answer marker; this is the whole '
                          'response.'}


def user_text(j):
    note = f'\n({NOTES[j["method"]]})' if j['method'] in NOTES else ''
    return (f'<question>\n{j["question"]}\n</question>\n\n<key_points>\n{fmt_points(j["points"])}\n</key_points>\n\n'
            f'<answer>{note}\n{j["answer"]}\n</answer>')


def jkey(j):
    return f'{j["config"]}/{j["file"]}/{j["id"]}'


def custom_id(key):
    return 'j' + hashlib.sha1(key.encode()).hexdigest()[:24]


def load_jobs(args):
    """All (config, file, item) judgments, or with --sample a seeded sample stratified over config x file."""
    questions = {it['id']: it['question'] for items in eval_sets(args.data, args.gen).values() for it in items}
    kps = {}
    for r in load(os.path.join(args.keypoints, 'keypoints.jsonl')):
        if r['status'] == 'ok':
            kps[r['id']] = r
    adir = os.path.join(args.scores, 'answers')
    configs = args.configs or sorted(d for d in os.listdir(adir) if not d.startswith('_'))
    strata = []
    for config in configs:
        for fname in sorted(os.listdir(os.path.join(adir, config))):
            file = fname.removesuffix('.jsonl')
            if args.files and file not in args.files:
                continue
            jobs = []
            for a in load(os.path.join(adir, config, fname)):
                if a['id'] not in kps:
                    sys.exit(f'{a["id"]}: no key points')
                jobs.append({'config': config, 'file': file, 'id': a['id'], 'question': questions[a['id']],
                             'points': kps[a['id']]['points'], 'answer': a['answer'], 'method': a['method'],
                             'finish_reason': a['finish_reason']})
            strata.append(jobs)
    if not args.sample:
        return [j for s in strata for j in s]
    rng = random.Random(SAMPLE_SEED)
    for s in strata:
        rng.shuffle(s)
    out = []  # round-robin over strata, so any prefix of the sample covers every config x file
    for k in range(max(map(len, strata))):
        out += [s[k] for s in strata if k < len(s)]
    return out[:args.sample]


# ---------- scoring ----------

PRECISION_MAX_REL = 0.005  # the precision rule applies to points at the default tolerance or tighter
WRITTEN_RE = re.compile(rf'({NUM})(?:\s*(?:\\times|×|x|\*)\s*10\s*\^\s*\{{?\s*([-+]?\d+)\s*\}}?|[eE]([-+]?\d+))?')


def written(s):
    """(value, step) for every number written in s; step = one unit in its last significant digit
    (trailing zeros of a whole number are not significant: "1200" -> 100)."""
    out = []
    for m in WRITTEN_RE.finditer(s.replace('−', '-').replace('$', '')):
        mant, e = m[1].replace(',', ''), int(m[2] or m[3] or 0)
        if '.' in mant:
            step = 10.0 ** -len(mant.split('.')[1])
        else:
            digits = mant.lstrip('0')
            step = 10.0 ** (len(digits) - len(digits.rstrip('0'))) if digits else 1.0
        out.append((float(mant) * 10.0 ** e, step * 10.0 ** e))
    return out


def written_step(text, x):
    """Step of the number in text that is x (up to a power of 1000, for unit prefixes), else None."""
    for n, step in written(text or ''):
        for e in range(-4, 5):
            if n and math.isclose(abs(x), n * 1000.0 ** e, rel_tol=2e-3):
                return step * 1000.0 ** e
            if not n and x == 0:
                return step
    return None


def close(x, v, p, quote=None):
    tol = max(p['rel_tol'] * abs(v['value']), p['abs_tol'], 1e-9 * max(1, abs(v['value'])))
    d = abs(x - v['value'])
    if v['unit'] in ('degrees', '°', 'deg'):
        d = min(d % 360, 360 - d % 360)
    if d > tol:
        return False
    if p['rel_tol'] > PRECISION_MAX_REL or (p['rel_tol'] == 0 and p['abs_tol'] == 0):
        return True
    step = written_step(quote, x)
    if step is None:  # value not written in the quote (derived by the judge): tolerance only
        return True
    # both are roundings of the same true value: allow half a step of each
    step += written_step(v.get('as_written', ''), v['value']) or 0
    return d <= 0.5 * step * (1 + 1e-9)


def numeric_ok(p, values, quote=None):
    """Polar points (a magnitude unit plus degrees) need both parts; otherwise any alternative."""
    got = [(unit_label(u['unit']), u['value']) for u in values]  # judge may return '' for a unitless value

    def match(alts):
        return any(close(x, v, p, quote) for v in alts for unit, x in got if unit == unit_label(v['unit']))
    deg = [v for v in p['values'] if v['unit'] == 'degrees']
    mag = [v for v in p['values'] if v['unit'] != 'degrees']
    if deg and mag:
        return match(deg) and match(mag)
    return match(p['values'])


def point_label(p, res):
    """covered / partly / wrong / missing for one judged point."""
    v = res['verdict']
    if p['kind'] == 'numeric':
        if v == 'stated':
            return 'covered' if numeric_ok(p, res['values'], res['quote']) else 'wrong'
        return 'wrong' if v == 'conflicting' else 'missing'
    return {'stated': 'covered', 'conflicting': 'wrong'}.get(v, v)


SCORE = {'covered': 1.0, 'partly': 0.5, 'wrong': 0.0, 'missing': 0.0}


def labels(rec, points):
    return [point_label(p, res) for p, res in zip(points, rec['points'])]


def quote_check(rec, points):
    """Numeric points whose reported value doesn't appear in the quote (up to a power of 1000)."""
    bad = []
    for k, (p, res) in enumerate(zip(points, rec['points']), 1):
        if p['kind'] != 'numeric' or res['verdict'] != 'stated':
            continue
        nums = reference_numbers(res['quote'])
        for u in res['values']:
            x = abs(u['value'])
            if not any(math.isclose(x, n * 1000.0 ** e, rel_tol=2e-3) for n in nums for e in range(-4, 5)) \
                    and not (x == 0 and 0 in nums):
                bad.append(k)
    return bad


# ---------- API ----------

def request_params(j, model, effort):
    return {
        'model': model,
        'max_tokens': MAX_TOKENS,
        'system': [{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
        'thinking': {'type': 'adaptive'},
        'output_config': {'effort': effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        'messages': [{'role': 'user', 'content': user_text(j)}],
    }


def cost(model, u, batch=False):
    p = PRICES[model]
    c = (u['input'] * p['in'] + u['output'] * p['out'] + u['cache_read'] * p['cache_read']
         + u['cache_write'] * p['cache_write']) / 1e6
    return c / 2 if batch else c


def answer_sha(j):
    return hashlib.sha1(j['answer'].encode()).hexdigest()[:16]


def base_rec(j, model, effort):
    return {'key': jkey(j), 'config': j['config'], 'file': j['file'], 'id': j['id'], 'model': model,
            'effort': effort, 'prompt_version': PROMPT_VERSION, 'answer_sha': answer_sha(j)}


def parse_message(rec, msg, n_points):
    u = msg.usage
    rec.update(stop_reason=msg.stop_reason,
               usage={'input': u.input_tokens, 'output': u.output_tokens,
                      'cache_read': u.cache_read_input_tokens or 0,
                      'cache_write': u.cache_creation_input_tokens or 0})
    if msg.stop_reason != 'end_turn':
        return {**rec, 'status': 'error', 'error': f'stop_reason {msg.stop_reason}'}
    return parse_points(rec, next((b.text for b in msg.content if b.type == 'text'), ''), n_points)


def parse_points(rec, text, n_points):
    try:
        pts = json.loads(text)['points']
    except (json.JSONDecodeError, KeyError):
        return {**rec, 'status': 'error', 'error': 'invalid JSON', 'raw': text}
    if [r['point'] for r in pts] != list(range(1, n_points + 1)):
        return {**rec, 'status': 'error', 'error': f'points {[r["point"] for r in pts]} != 1..{n_points}', 'raw': text}
    rec.update(points=pts, status='ok')
    return rec


def judge_one(client, j, model, effort):
    import anthropic
    rec = base_rec(j, model, effort)
    try:
        with client.messages.stream(**request_params(j, model, effort)) as stream:
            msg = stream.get_final_message()
            rec['request_id'] = stream.request_id
    except anthropic.APIStatusError as e:
        return {**rec, 'status': 'error', 'error': f'{e.status_code}: {e.message}'}
    except anthropic.APIConnectionError as e:
        return {**rec, 'status': 'error', 'error': f'connection: {e}'}
    return parse_message(rec, msg, len(j['points']))


# ---------- local judge (llama-server) ----------

def local_payload(j, think):
    return {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_text(j)}],
            'temperature': 0.0, 'top_k': 1, 'seed': 0, 'max_tokens': MAX_TOKENS, 'cache_prompt': True,
            'chat_template_kwargs': {'enable_thinking': think},
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'grades', 'schema': SCHEMA}}}


def judge_local(args, model, j):
    import urllib.error
    rec = {**base_rec(j, model, 'think' if args.think else 'no_think'), 'backend': 'llama-server'}
    try:
        res = request(args.url, '/v1/chat/completions', local_payload(j, args.think), args.timeout)
    except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as e:
        return {**rec, 'status': 'error', 'error': str(e)}
    ch, u = res['choices'][0], res.get('usage', {})
    rec.update(stop_reason=ch.get('finish_reason'),
               usage={'input': u.get('prompt_tokens'), 'output': u.get('completion_tokens')})
    if ch.get('finish_reason') != 'stop':
        return {**rec, 'status': 'error', 'error': f'finish_reason {ch.get("finish_reason")}'}
    return parse_points(rec, ch['message'].get('content') or '', len(j['points']))


def out_path(out_dir):
    return os.path.join(out_dir, 'judgments.jsonl')


def latest_ok(out_dir):
    """{key: latest ok record of the current prompt version}."""
    if not os.path.exists(out_path(out_dir)):
        return {}
    return {r['key']: r for r in load(out_path(out_dir))
            if r['status'] == 'ok' and r['prompt_version'] == PROMPT_VERSION}


def client_():
    import anthropic
    return anthropic.Anthropic(max_retries=4)


def cmd_estimate(args):
    jobs = load_jobs(args)
    model = JUDGES[args.judge]
    sys_tok = len(SYSTEM) / 3.5
    inp = sum(len(user_text(j)) / 3.5 for j in jobs)
    n = len(jobs)
    files = {}
    for j in jobs:
        files[(j['config'], j['file'])] = files.get((j['config'], j['file']), 0) + 1
    print(f'{n} judgments ({len(files)} config x file), {model}, effort {args.effort}, prompt {PROMPT_VERSION}')
    print(f'input ~{inp / n:,.0f} tokens/judgment + system ~{sys_tok:,.0f} (rough: chars / 3.5)')
    for out in (args.out_low, args.out_high):
        u = {'input': inp + sys_tok * n, 'output': out * n, 'cache_read': 0, 'cache_write': 0}
        print(f'  at {out} output tokens/judgment: standard ${cost(model, u):,.2f}, batch ${cost(model, u, True):,.2f}')


def cmd_run(args):
    if args.judge == 'local':
        if not args.url:
            sys.exit('--judge local needs --url')
        models = server_info(args.url, args.timeout)['models']
        if not isinstance(models, list) or len(models) != 1:
            sys.exit(f'server not reachable or not one model: {models}\n(--url needs http://; a 401 means '
                     f'LLAMA_API_KEY is missing or wrong)')
        model = models[0]
        call = lambda j: judge_local(args, model, j)  # noqa: E731
        effort = 'think' if args.think else 'no_think'
    else:
        client = client_()
        model = JUDGES[args.judge]
        call = lambda j: judge_one(client, j, model, args.effort)  # noqa: E731
        effort = args.effort
    os.makedirs(args.out, exist_ok=True)
    done = latest_ok(args.out)
    if any(r['model'] != model for r in done.values()):
        sys.exit(f'{args.out} holds judgments by another model; use another --out')
    redo = set()
    if args.redo:
        with open(args.redo) as f:
            redo = {line.strip() for line in f if line.strip()}
    # a key listed in --redo counts as done only once a judgment of its current answer exists (resumable)
    todo = [j for j in load_jobs(args) if jkey(j) not in done
            or (jkey(j) in redo and done[jkey(j)].get('answer_sha') != answer_sha(j))]
    if redo:
        print(f'--redo: {len(redo)} keys listed, {sum(jkey(j) in redo for j in todo)} still to re-judge')
    print(f'{len(todo)} judgments to do ({len(done)} done), {model}, effort {effort}, prompt {PROMPT_VERSION}')
    total = 0.0
    with open(out_path(args.out), 'a') as f, cf.ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(call, j) for j in todo]
        for k, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            f.write(json.dumps(rec) + '\n')  # save before anything else can fail
            f.flush()
            if 'usage' in rec and model in PRICES:
                total += cost(model, rec['usage'])
            if rec['status'] != 'ok' or k % 20 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} {rec['key']} {rec['status']}"
                      + (f" ({rec['error']})" if rec['status'] != 'ok' else '') + f'  running cost ${total:.2f}')


def batches_path(out_dir):
    return os.path.join(out_dir, 'batches.jsonl')


def load_batches(out_dir):
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
    model = JUDGES[args.judge]
    os.makedirs(args.out, exist_ok=True)
    done = set(latest_ok(args.out))
    pending = {k for b in load_batches(args.out).values() if not b.get('collected') for k in b['keys']}
    todo = [j for j in load_jobs(args) if jkey(j) not in done | pending]
    print(f'{len(todo)} judgments to submit ({len(done)} done, {len(pending)} in uncollected batches), '
          f'{model}, effort {args.effort}, prompt {PROMPT_VERSION}')
    if not todo:
        return
    cids = [custom_id(jkey(j)) for j in todo]
    assert len(set(cids)) == len(cids)
    requests = [{'custom_id': c, 'params': request_params(j, model, args.effort)} for c, j in zip(cids, todo)]
    size = len(json.dumps(requests))
    assert len(requests) <= 100_000 and size < 200e6, (len(requests), size)  # API limits: 100k / 256 MB
    batch = client.messages.batches.create(requests=requests)
    log_batch(args.out, {'batch_id': batch.id, 'model': model, 'effort': args.effort,
                         'prompt_version': PROMPT_VERSION, 'keys': [jkey(j) for j in todo]})
    print(f'submitted {batch.id}: {len(requests)} requests, {size / 1e6:.1f} MB')


def cmd_status(args):
    client = client_()
    for bid, b in load_batches(args.out).items():
        if b.get('collected'):
            print(f'{bid} collected ({len(b["keys"])} judgments)')
            continue
        info = client.messages.batches.retrieve(bid)
        n = info.request_counts
        print(f'{bid} {info.processing_status}: processing {n.processing}, succeeded {n.succeeded}, '
              f'errored {n.errored}, canceled {n.canceled}, expired {n.expired}')


def cmd_collect(args):
    client = client_()
    by_cid = {custom_id(jkey(j)): j for j in load_jobs(args)}
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
                j = by_cid[res.custom_id]
                rec = base_rec(j, b['model'], b['effort'])
                rec.update(batch_id=bid, batch=True, prompt_version=b['prompt_version'])
                t = res.result.type
                if t == 'succeeded':
                    rec = parse_message(rec, res.result.message, len(j['points']))
                    total += cost(b['model'], rec['usage'], batch=True)
                elif t == 'errored':
                    e = res.result.error.error
                    rec.update(status='error', error=f'{e.type}: {e.message}')
                else:  # canceled / expired: not billed, resubmit
                    rec.update(status='error', error=t)
                f.write(json.dumps(rec) + '\n')
                counts[rec['status']] = counts.get(rec['status'], 0) + 1
        log_batch(args.out, {'batch_id': bid, 'collected': True})
        print(f'{bid} collected: {counts}, cost ${total:.2f}')


# ---------- hand grades ----------

def fmt_point_human(k, p):
    if p['kind'] == 'numeric':
        vals = ' | '.join(f'{v["as_written"]} {v["unit"]}'.strip() for v in p['values'])
        tol = f'±{p["rel_tol"]:.1%}' + (f', abs ±{p["abs_tol"]:g}' if p['abs_tol'] else '')
        return f'  {k}. [numeric] {p["label"]} = {vals} ({tol})'
    if p['kind'] == 'exact':
        return f'  {k}. [exact] {p["label"]}: {p["statement"]}'
    return f'  {k}. [conceptual] {p["label"]}: {p["statement"]}'


def cmd_grade(args):
    jobs = load_jobs(args)[:args.n]
    done = {r['key'] for r in load(args.out)} if os.path.exists(args.out) else set()
    todo = [j for j in jobs if jkey(j) not in done]
    print(f'{len(todo)} answers to grade ({len(done)} done). Per key point: '
          'c = covered, p = partly, n = not (wrong or missing); s = skip answer, '
          'f = flag the item with a note (not graded, listed by `agree`), q = quit.\n'
          'Grade the final answer only, not the working behind it.\n'
          'Numeric: covered only if the value is within the tolerance shown and, at the default tolerance '
          '(<= 0.5%), is the key rounded to the digits written (0.662 or 0.66 for 0.661619, not 0.664).')
    for i, j in enumerate(todo, 1):
        print('\n' + '=' * 78 + f'\n[{i}/{len(todo)}] {jkey(j)}\n')
        print('QUESTION:\n' + j['question'].strip() + '\n')
        print('KEY POINTS:\n' + '\n'.join(fmt_point_human(k, p) for k, p in enumerate(j['points'], 1)) + '\n')
        print('ANSWER' + (f' ({NOTES[j["method"]]})' if j['method'] in NOTES else '')
              + ':\n' + j['answer'].strip() + '\n')
        grades = []
        for k, p in enumerate(j['points'], 1):
            while True:
                g = input(f'  point {k} ({p["label"]}) [c/p/n/s/f/q]: ').strip().lower()
                if g in ('c', 'p', 'n', 's', 'f', 'q'):
                    break
            if g in ('s', 'f', 'q'):
                break
            grades.append({'c': 'covered', 'p': 'partly', 'n': 'not'}[g])
        if g == 'q':
            break
        if g == 's':
            continue
        rec = {'key': jkey(j), 'labels': grades}
        if g == 'f':
            rec = {'key': jkey(j), 'flag': input('  note: ').strip()}
        with open(args.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    print(f'\nsaved to {args.out}')


# ---------- agreement ----------

def coarse(label):
    """3 classes for comparisons with hand grades: covered / partly / not."""
    return label if label in ('covered', 'partly') else 'not'


def kappa(pairs, classes=('covered', 'partly', 'not')):
    n = len(pairs)
    if not n:
        return float('nan')
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in classes)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def compare(name, la, lb, kinds):
    """la, lb: {key: [coarse label per point]} -> report lines."""
    keys = sorted(set(la) & set(lb))
    pairs = [(a, b, k) for key in keys for a, b, k in zip(la[key], lb[key], kinds[key])]
    lines = [f'### {name}: {len(keys)} answers, {len(pairs)} points', '',
             '| points | n | agree | kappa |', '|---|---|---|---|']
    for kind in ('all', 'numeric', 'exact', 'conceptual'):
        sel = [(a, b) for a, b, k in pairs if kind == 'all' or k == kind]
        if sel:
            lines.append(f'| {kind} | {len(sel)} | {sum(a == b for a, b in sel) / len(sel):.1%} | {kappa(sel):.2f} |')
    sa = [sum(SCORE.get(x, 0) for x in la[k]) / len(la[k]) for k in keys]
    sb = [sum(SCORE.get(x, 0) for x in lb[k]) / len(lb[k]) for k in keys]
    if keys:
        lines += ['', f'Item score: mean {sum(sa) / len(sa):.3f} vs {sum(sb) / len(sb):.3f}, '
                  f'mean abs difference {sum(abs(a - b) for a, b in zip(sa, sb)) / len(keys):.3f}; '
                  f'all points covered: {sum(all(x == "covered" for x in la[k]) for k in keys)} vs '
                  f'{sum(all(x == "covered" for x in lb[k]) for k in keys)}', '']
    return lines


def cmd_agree(args):
    jobs = {jkey(j): j for j in load_jobs(args)}
    recs = {}
    for side in (args.a, args.b):
        r = {k: v for k, v in latest_ok(side).items() if k in jobs}
        recs[side] = r
    full = {side: {k: labels(r, jobs[k]['points']) for k, r in recs[side].items()} for side in recs}
    kinds = {k: [p['kind'] for p in j['points']] for k, j in jobs.items()}
    name = {side: os.path.basename(os.path.normpath(side)) for side in recs}
    na, nb = name[args.a], name[args.b]
    out = [f'# Judge agreement ({len(jobs)} sampled answers, seed {SAMPLE_SEED})', '']
    for side in recs:
        rs = list(recs[side].values())
        spent = sum(cost(r['model'], r['usage'], r.get('batch', False)) for r in rs if r['model'] in PRICES)
        outs = sorted(r['usage']['output'] for r in rs) or [0]
        qbad = sum(len(quote_check(r, jobs[r['key']]['points'])) for r in rs)
        out.append(f'- {name[side]}: {len(rs)} ok judgments, ${spent:.3f}, output tokens median '
                   f'{outs[len(outs) // 2]:,} max {outs[-1]:,}; numeric values not found in their quote: {qbad}')
    out.append('')
    ca = {k: [coarse(x) for x in v] for k, v in full[args.a].items()}
    cb = {k: [coarse(x) for x in v] for k, v in full[args.b].items()}
    out += compare(f'{na} vs {nb}', ca, cb, kinds)
    hum = {}
    if args.human:
        last = {r['key']: r for r in load(args.human) if r['key'] in jobs}  # latest record per key wins
        flags = {k: r['flag'] for k, r in last.items() if 'flag' in r}
        hum = {k: r['labels'] for k, r in last.items() if 'labels' in r and len(r['labels']) == len(jobs[k]['points'])}
        out += compare(f'human vs {na}', hum, ca, kinds)
        out += compare(f'human vs {nb}', hum, cb, kinds)
        if flags:
            out += [f'### Flagged by the grader (not graded): {len(flags)}', '']
            out += [f'- {k} — {note or "(no note)"}' for k, note in sorted(flags.items())] + ['']
    out += ['## Disagreements', '', f'Labels per point: {na} / {nb}' + (' / human' if args.human else '')
            + '. Numeric values are what the judge read from the answer.', '']
    for k in sorted(set(full[args.a]) & set(full[args.b])):
        j = jobs[k]
        diff = [i for i, (a, b) in enumerate(zip(ca[k], cb[k])) if a != b]
        diff += [i for i, h in enumerate(hum.get(k, [])) if h != ca[k][i] and i not in diff]
        if not diff:
            continue
        ra, rb = recs[args.a][k]['points'], recs[args.b][k]['points']
        q = ' '.join(j['question'].split())
        out += [f'### {k}', '', f'**Q:** {q[:300]}' + ('…' if len(q) > 300 else ''), '',
                '**Answer:** ' + ' '.join(j['answer'].split())[:800], '']
        for i in sorted(diff):
            p = j['points'][i]
            def show(r, lab):
                v = ', '.join(f'{u["value"]:g} {u["unit"]}' for u in r[i]['values'])
                return f'{lab}' + (f' ({v})' if v else '') + (f' — {r[i]["note"]}' if r[i]['note'] else '')
            out.append(f'- {fmt_point_human(i + 1, p).strip()}' + (f' — {p["statement"]}' if p['kind'] == 'numeric' else ''))
            out.append(f'  - {na}: {show(ra, full[args.a][k][i])}')
            out.append(f'  - {nb}: {show(rb, full[args.b][k][i])}')
            if k in hum:
                out.append(f'  - human: {hum[k][i]}')
        out += ['', '---']
    path = os.path.join(os.path.dirname(os.path.normpath(args.a)), f'agree_{na}_vs_{nb}.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out[:out.index('## Disagreements')]))
    print(f'wrote {path}')


# ---------- report ----------

# pool-less solo exam groups merged into one group per family for scoring
GROUP_MERGE = {**{f'vg_solo_{n}': 'vg_password' for n in range(1398, 1406)},
               **{f'vg_solo_{n}': 'vg_readings' for n in range(2029, 2034)}}
BOOT_N = 2000
BOOT_SEED = 11


def clusters(args):
    """{item id: cluster}: merged variant group (exam) or source chunk (generated questions)."""
    out = {}
    for items in eval_sets(args.data, args.gen).values():
        for it in items:
            vg = it.get('variant_group')
            out[it['id']] = GROUP_MERGE.get(vg, vg) if vg else it['chunk_id']
    return out


def summarize(vals, cl, macro):
    """vals {id: x}; macro: mean of cluster means (exam headline), else item mean.
    Returns (estimate, lo, hi) with a 95% bootstrap interval over clusters."""
    by = {}
    for i, x in vals.items():
        by.setdefault(cl[i], []).append(x)
    groups = list(by.values())
    if not groups:
        return None

    def est(gs):
        if macro:
            return sum(sum(g) / len(g) for g in gs) / len(gs)
        return sum(map(sum, gs)) / sum(map(len, gs))
    rng = random.Random(BOOT_SEED)
    boots = sorted(est([rng.choice(groups) for _ in groups]) for _ in range(BOOT_N))
    return est(groups), boots[int(0.025 * BOOT_N)], boots[int(0.975 * BOOT_N) - 1]


def fmt_ci(r, signed=False):
    if r is None:
        return '—'
    f = '{:+.3f}' if signed else '{:.3f}'
    return f'{f.format(r[0])} [{f.format(r[1])}, {f.format(r[2])}]'


def cmd_report(args):
    jobs = {jkey(j): j for j in load_jobs(args)}
    recs = {k: r for k, r in latest_ok(args.out).items() if k in jobs}
    cl = clusters(args)
    item, allok, kinds_ok = {}, {}, {}  # (config, file) -> {id: value}
    no_points = set()
    for k, j in jobs.items():
        if not j['points']:
            no_points.add(j['id'])
            continue
        if k not in recs:
            continue
        lab = labels(recs[k], j['points'])
        cf = (j['config'], j['file'])
        item.setdefault(cf, {})[j['id']] = sum(SCORE[x] for x in lab) / len(lab)
        allok.setdefault(cf, {})[j['id']] = float(all(x == 'covered' for x in lab))
        for p, x in zip(j['points'], lab):
            kinds_ok.setdefault(cf, {}).setdefault(p['kind'], []).append(SCORE[x])
    cfs = sorted({(j['config'], j['file']) for j in jobs.values()})
    n_job = {cf: sum((j['config'], j['file']) == cf and bool(j['points']) for j in jobs.values()) for cf in cfs}
    macro = lambda f: f.startswith('exam')
    out = [f'# Scores ({os.path.basename(os.path.normpath(args.out))}, prompt {PROMPT_VERSION})', '',
           f'{len(recs)} of {len(jobs)} answers judged; {len(no_points)} items have no gradable key points (skipped). '
           'Item score = mean over key points (covered 1, partly 0.5, else 0). **Exam: macro average over variant '
           f'groups** (solo families merged: {len(set(cl[i] for i in cl if i.startswith("xq_")))} groups), '
           'Fable: item mean. 95% bootstrap intervals over groups (exam) or source chunks (Fable), '
           f'{BOOT_N} resamples.', '',
           '## Score per config and file', '',
           '| config | file | judged | score | all points correct | numeric | exact | conceptual |',
           '|' + '---|' * 8]
    for cf in cfs:
        kc = kinds_ok.get(cf, {})
        kind = lambda kd: f'{sum(kc[kd]) / len(kc[kd]):.3f} (n {len(kc[kd])})' if kc.get(kd) else '—'
        out.append(f'| {cf[0]} | {cf[1]} | {len(item.get(cf, {}))}/{n_job[cf]} | '
                   f'{fmt_ci(summarize(item.get(cf, {}), cl, macro(cf[1])))} | '
                   f'{fmt_ci(summarize(allok.get(cf, {}), cl, macro(cf[1])))} | '
                   f'{kind("numeric")} | {kind("exact")} | {kind("conceptual")} |')
    out += ['', 'Point-kind columns are pooled over points (no group weighting).', '',
            '## With documents vs without (paired, same items)', '',
            'Difference = top5 − nodocs item score, aggregated as above. Better / worse / same = items whose score '
            'went up / down / stayed.', '',
            '| config | set | items | top5 − nodocs | better | worse | same |', '|' + '---|' * 7]
    for config in sorted({c for c, _ in cfs}):
        for st in sorted({f.split('_')[0] for _, f in cfs}):
            a, b = item.get((config, f'{st}_top5'), {}), item.get((config, f'{st}_nodocs'), {})
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            d = {i: a[i] - b[i] for i in ids}
            out.append(f'| {config} | {st} | {len(ids)} | {fmt_ci(summarize(d, cl, st == "exam"), signed=True)} | '
                       f'{sum(x > 0 for x in d.values())} | {sum(x < 0 for x in d.values())} | '
                       f'{sum(x == 0 for x in d.values())} |')
    configs = sorted({c for c, _ in cfs})
    out += ['', f'## Each config vs {args.base} (paired, same items)', '',
            f'| config | file | items | config − {args.base} | better | worse | same |', '|' + '---|' * 7]
    for st in sorted({f.split('_')[0] for _, f in cfs}):
        for cond in ('nodocs', 'top5'):
            base = item.get((args.base, f'{st}_{cond}'), {})
            for config in configs:
                if config == args.base:
                    continue
                a = item.get((config, f'{st}_{cond}'), {})
                ids = sorted(set(a) & set(base))
                if not ids:
                    continue
                d = {i: a[i] - base[i] for i in ids}
                out.append(f'| {config} | {st}_{cond} | {len(ids)} | '
                           f'{fmt_ci(summarize(d, cl, st == "exam"), signed=True)} | {sum(x > 0 for x in d.values())} | '
                           f'{sum(x < 0 for x in d.values())} | {sum(x == 0 for x in d.values())} |')
    if args.grounding:
        out += ['', '## Quote grounding (top5; from `score.py ground`)', '',
                '| config | file | items with quotes | quotes not found (none) | items with a "none" quote | '
                'score, items with a "none" quote | score, other items |', '|' + '---|' * 7]
        for cf in cfs:
            path = os.path.join(args.grounding, cf[0], cf[1] + '.jsonl')
            if 'nodocs' in cf[1] or not os.path.exists(path):
                continue
            g = {r['id']: r for r in load(path)}
            q = [r for r in g.values() if r['n']]
            n = sum(r['n'] for r in q)
            nn = sum(r['none'] for r in q)
            sc = item.get(cf, {})
            bad = [sc[i] for i in sc if g.get(i, {}).get('none')]
            ok = [sc[i] for i in sc if not g.get(i, {}).get('none')]
            m = lambda xs: f'{sum(xs) / len(xs):.3f} (n {len(xs)})' if xs else '—'
            out.append(f'| {cf[0]} | {cf[1]} | {len(q)} | {nn} of {n} ({nn / n:.0%}) | {sum(1 for r in q if r["none"])} | '
                       f'{m(bad)} | {m(ok)} |' if n else f'| {cf[0]} | {cf[1]} | 0 | — | — | — | {m(ok)} |')
        out += ['', 'Score columns are plain item means (no group weighting).']
    path = os.path.join(args.out, 'report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('estimate', 'run', 'submit', 'status', 'collect', 'grade', 'agree', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='data/<source> folder')
        p.add_argument('--gen', required=True, help='generate.py output dir of the held-out questions, relative to --data')
        p.add_argument('--scores', required=True, help='score.py output folder (holds answers/<config>/<file>.jsonl)')
        p.add_argument('--keypoints', required=True, help='keypoints.py output folder')
        p.add_argument('--configs', nargs='+', help='config folders (default: all not starting with "_")')
        p.add_argument('--files', nargs='+', help='eval files, e.g. exam_top5 (default: all)')
        p.add_argument('--sample', type=int, default=0, help='seeded sample of N answers, stratified over config x file')
        if name in ('estimate', 'run', 'submit', 'status', 'collect'):
            p.add_argument('--judge', default='haiku', choices=sorted(JUDGES) + (['local'] if name == 'run' else []))
            p.add_argument('--effort', default='medium', choices=['low', 'medium', 'high'])
        if name in ('run', 'submit', 'status', 'collect', 'grade', 'report'):
            p.add_argument('--out', required=True)
        if name == 'report':
            p.add_argument('--grounding', help='score.py ground output folder (grounding/<config>/<file>.jsonl)')
            p.add_argument('--base', default='e4b_base', help='config the others are compared with')
        if name == 'estimate':
            p.add_argument('--out-low', type=int, default=300, help='assumed output tokens per judgment, low')
            p.add_argument('--out-high', type=int, default=1500, help='assumed output tokens per judgment, high')
        if name == 'run':
            p.add_argument('--workers', type=int, default=8, help='with --judge local: match the server --parallel')
            p.add_argument('--url', help='llama-server base URL (--judge local)')
            p.add_argument('--think', action='store_true', help='--judge local: thinking on (default off)')
            p.add_argument('--timeout', type=float, default=900, help='--judge local: seconds per request')
            p.add_argument('--redo', help='file of judgment keys (config/file/id, one per line) to judge again when '
                                          'their extracted answer changed; the newest ok judgment counts')
        if name == 'grade':
            p.add_argument('--n', type=int, default=20, help='grade the first N answers of the sample')
        if name == 'agree':
            p.add_argument('--a', required=True, help='judge output folder')
            p.add_argument('--b', required=True, help='second judge output folder')
            p.add_argument('--human', help='hand grades from `grade`')
    args = ap.parse_args()
    {'estimate': cmd_estimate, 'run': cmd_run, 'submit': cmd_submit, 'status': cmd_status, 'collect': cmd_collect,
     'grade': cmd_grade, 'agree': cmd_agree, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
