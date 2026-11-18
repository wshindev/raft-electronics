#!/usr/bin/env python3
"""Step 3: describe rendered figures with Claude vision.

Each request sends the PNG, the text hints extracted from the drawing file
(image_meta.jsonl) and the surrounding text (image_refs.jsonl), and asks for a
JSON description (see SCHEMA). Exam-question figures get the answer-leak rule.

Usage (pilot):
    # offline cost estimate, no API calls
    uv run python3 -I pipeline/describe.py estimate --refs data/model/image_refs.jsonl \
        --images data/model/pilot/pilot_images.jsonl

    # describe with each model (synchronous calls; re-runs skip finished images)
    uv run python3 -I pipeline/describe.py run --refs data/model/image_refs.jsonl \
        --images data/model/pilot/pilot_images.jsonl --out data/model/pilot

    # side-by-side HTML for review + measured cost and full-run extrapolation
    uv run python3 -I pipeline/describe.py review --refs data/model/image_refs.jsonl \
        --images data/model/pilot/pilot_images.jsonl --out data/model/pilot

--meta defaults to image_meta.jsonl next to --refs, PNGs to png/ next to --refs.
`run` needs the `anthropic` package and ANTHROPIC_API_KEY in the environment.
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import html
import json
import os
import sys
import threading

MODELS = ['claude-sonnet-5-5', 'claude-opus-5-5']

# USD per million tokens, standard (non-batch) rates. Batch API = 50% of these.
# Source: claude-api skill model table, cached 2026-10-06. Re-check before a full run.
PRICES = {
    'claude-sonnet-5-5': {'in': 2.00, 'out': 10.00, 'cache_read': 0.20, 'cache_write': 2.50},
    'claude-opus-5-5': {'in': 4.00, 'out': 20.00, 'cache_read': 0.20, 'cache_write': 5.00},
}

MAX_USES = 3        # surrounding-text snippets per image
MAX_TOKENS = 16000  # output cap (thinking + JSON); well above expected use

SYSTEM = """\
You write descriptions of technical figures from an electronics curriculum. Each \
description will replace its figure in text used to train and evaluate language \
models, so a reader who never sees the image must be able to answer questions that \
depend on it. Accuracy matters more than brevity: a wrong value or connection is \
worse than a gap you flag as uncertain.

You receive:
1. The rendered figure.
2. Text hints extracted from the figure's drawing file, when available: visible \
labels and the library symbols placed (e.g. "resistor x4"). Use them for the exact \
spelling of labels and values you can see; subscripts are written with "_" (V_out). \
They can include text that isn't visible and can miss text drawn inside symbols \
(instrument dials, chip pins), so the image decides what is actually shown. \
Symbol names come from the drawing library and may be generic ("dot" is a junction, \
"arrow" may be annotation). Where a symbol name encodes a device type or polarity \
(npn/pnp, mosfet_n*/mosfet_p* for N-/P-channel, nc_*/no_* for normally closed/open \
contacts), trust it over your reading of small arrows, bubbles or contact gaps in \
the pixels; the hint gives counts, not positions, so match each symbol to its place \
in the drawing. If the image seems to contradict a symbol name, say so in "uncertain".
3. Text that surrounds the figure where it is used (LaTeX source, truncated). Use it \
to understand what the figure is for and what the author calls things. Don't add \
facts from the text that the figure itself doesn't show.
4. The role of the figure, which sets what you may state (see the instructions in \
the request).

Fields:
- figure_type: the closest category. Single-line (one-line) power diagrams are \
schematics.
- text_in_figure: every piece of text visible in the figure, verbatim, in reading \
order (labels, values, pin names, dial markings, axis ticks, annotations).
- components: every component that has terminals in "connections", once each: id \
(the same ID used in connections and the description), type (the specific device, \
with polarity or variant where it matters: "P-channel enhancement MOSFET", "NPN \
transistor", "normally-closed pushbutton", "electrolytic capacitor", "voltmeter"), \
and value or rating as printed ("4.7 kΩ", "12 V", "" if none). Empty when \
"connections" is empty.
- connections: required whenever wires or conductors are drawn between components, \
whatever the figure type: schematics, single-line power diagrams, ladder diagrams, \
pictorial wiring, instrument hookups (probe tip and ground clip are terminals), and \
small circuits drawn inside plots or insets. One entry per electrical node, listing \
every terminal on that node. In a single-line diagram each drawn line stands for all \
phases; treat it as one node and say so in the description. All ground symbols form \
one node named "ground" unless the figure distinguishes them (e.g. separate chassis \
and signal grounds, or an isolated ground). Name terminals by component \
and pin, e.g. "R1.1", "Q1.collector", "U1.pin 7 (RST)", "V1.+", "lamp L1.left". Use \
designators shown in the figure; for unlabeled parts assign a short ID that includes \
the value or position, e.g. "R_4k7", "C_left", and use the same IDs in the \
description. Name a node after its label if it has one (+12 V, ground, V_out). For \
large regular arrays (memory matrices, PLAs), describe the repeating pattern once \
instead of listing every instance. If you infer a connection you can't trace (a \
line hidden behind another, an ambiguous crossing without a dot), list it and note \
it in "uncertain". Leave the list empty only when no wiring is drawn.
- description: complete, precise prose in plain text (Unicode symbols such as Ω, μ, \
°, no LaTeX). Cover what a technician would need: every component with its label and \
value, the topology, instrument settings (volts/div, time/div, coupling, trigger, \
selected ranges), waveform shapes with amplitudes, periods, offsets and phase \
relationships read from the graticule, plot axes with units and ranges and the shape \
and key points of each curve, and for photos what the objects are, their visible \
markings and the details that matter for the surrounding text. Spatial words \
(left, top) are fine when they help the reader follow the figure.
- summary: one sentence saying what the figure is.
- uncertain: anything you could not read or are inferring rather than seeing, each as \
a short note. Empty if nothing.
"""

ROLE_TEXT = {
    'question': """\
Role: EXAM QUESTION FIGURE. Students must answer a question about this figure. \
Describe only what is drawn. Do not state anything the question asks the student to \
work out: no calculated values, no predicted readings or waveforms, no identification \
of faults or of which component is responsible, no verdict on whether a design \
works, no explanation of how the circuit behaves beyond what is printed in the \
figure. If the figure is meant to be completed by the student (unconnected \
components, blank axes, empty table cells), describe what is present, state what is \
left unconnected or blank, and do not suggest how to complete it. List in \
"connections" only the wires that are actually drawn.""",
    'answer': """\
Role: EXAM ANSWER FIGURE. This figure is (part of) the reference answer to the exam \
question shown in the text. Describe it completely, including what it shows as the \
answer: added wiring, resulting waveforms, marked values, and how it differs from the \
question's figure if the text makes that clear.""",
    'reference': """\
Role: TEACHING FIGURE from the curriculum text. Describe it completely, including \
what the figure itself illustrates (e.g. annotated behavior, labeled regions).""",
}

FIGURE_TYPES = ['schematic', 'wiring_or_pictorial', 'ladder_or_control_diagram',
                'block_diagram', 'instrument_screen', 'plot_or_graph', 'photo',
                'pcb_or_layout', 'mechanical_or_physical', 'math_or_geometry',
                'table_or_text', 'other']

SCHEMA = {
    'type': 'object',
    'properties': {
        'figure_type': {'type': 'string', 'enum': FIGURE_TYPES},
        'text_in_figure': {'type': 'array', 'items': {'type': 'string'}},
        'components': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'type': {'type': 'string'},
                    'value': {'type': 'string'},
                },
                'required': ['id', 'type', 'value'],
                'additionalProperties': False,
            },
        },
        'connections': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'node': {'type': 'string'},
                    'terminals': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['node', 'terminals'],
                'additionalProperties': False,
            },
        },
        'description': {'type': 'string'},
        'summary': {'type': 'string'},
        'uncertain': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['figure_type', 'text_in_figure', 'components', 'connections', 'description',
                 'summary', 'uncertain'],
    'additionalProperties': False,
}

PROMPT_VERSION = hashlib.sha256(
    json.dumps([SYSTEM, ROLE_TEXT, SCHEMA], sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def role(ref):
    """Exam-question use wins: one description per image, so it must be leak-safe."""
    if ref['in_exam_question']:
        return 'question'
    if ref['in_exam_answer']:
        return 'answer'
    return 'reference'


def pick_uses(ref, r):
    """Surrounding-text snippets to send. Question figures get only their exam
    question text: module text around the same figure may explain the result."""
    order = {'question': ['exam_question'],
             'answer': ['exam_answer', 'module'],
             'reference': ['module']}[r]
    seen, out = set(), []
    for kind in order:
        for u in ref['uses']:
            text = (u['before'], u['after'])
            if u['kind'] == kind and text not in seen:
                seen.add(text)
                out.append(u)
    return out[:MAX_USES]


def hints_text(meta):
    labels = meta.get('labels') or []
    comps = meta.get('components') or {}
    lines = [f"Figure kind (from file creator): {meta.get('image_kind', 'unknown')}"]
    if labels:
        uniq = list(dict.fromkeys(labels))
        lines.append('Labels: ' + ' | '.join(uniq))
    if comps:
        lines.append('Symbols placed: ' + ', '.join(f'{k} x{v}' for k, v in comps.items()))
    if len(lines) == 1:
        lines.append('No text hints available for this figure.')
    return '\n'.join(lines)


def user_text(ref, meta):
    r = role(ref)
    parts = [ROLE_TEXT[r], '', '<text_hints>', hints_text(meta), '</text_hints>']
    for i, u in enumerate(pick_uses(ref, r), 1):
        parts += ['', f'<context n="{i}" where="{u["kind"]}">',
                  u['before'].strip(), '[FIGURE]', u['after'].strip(), '</context>']
    return '\n'.join(parts)


def request_params(model, ref, meta, png_path, effort, max_tokens=MAX_TOKENS):
    with open(png_path, 'rb') as f:
        data = base64.standard_b64encode(f.read()).decode('ascii')
    return {
        'model': model,
        'max_tokens': max_tokens,
        'system': [{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
        'thinking': {'type': 'adaptive'},
        'output_config': {'effort': effort,
                          'format': {'type': 'json_schema', 'schema': SCHEMA}},
        'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
            {'type': 'text', 'text': user_text(ref, meta)},
        ]}],
    }


def load_inputs(args):
    base = os.path.dirname(args.refs)
    refs = {r['image']: r for r in load(args.refs)}
    meta = {m['image']: m for m in load(args.meta or os.path.join(base, 'image_meta.jsonl'))}
    png_dir = args.png_dir or os.path.join(base, 'png')
    if args.images:
        selection = load(args.images)
    else:
        selection = [{'image': i, 'category': ''} for i in refs]
    items = []
    for s in selection:
        img = s['image']
        png = os.path.join(png_dir, img + '.png')
        if img in meta and os.path.exists(png):
            items.append({**s, 'ref': refs[img], 'meta': meta[img], 'png': png})
        else:
            print(f'skip {img}: no PNG', file=sys.stderr)
    return items


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def est_image_tokens(meta):
    return meta['width'] * meta['height'] / 750  # same approximation as render_report


def est_text_tokens(text):
    return len(text) / 3.0  # conservative for LaTeX-heavy text on the newer tokenizer


def cost(model, inp, out, cache_read=0, cache_write=0, batch=False):
    p = PRICES[model]
    usd = (inp * p['in'] + out * p['out'] + cache_read * p['cache_read']
           + cache_write * p['cache_write']) / 1e6
    return usd / 2 if batch else usd


def usage_cost(model, u):
    return cost(model, u['input'], u['output'], u['cache_read'], u['cache_write'])


def cmd_estimate(args):
    items = load_inputs(args)
    sys_tok = est_text_tokens(SYSTEM + json.dumps(SCHEMA))
    inp = sum(est_image_tokens(it['meta']) + est_text_tokens(user_text(it['ref'], it['meta']))
              for it in items)
    n = len(items)
    print(f'{n} images, prompt {PROMPT_VERSION}')
    print(f'est. input per image: {inp / n:,.0f} tokens (+ ~{sys_tok:,.0f} cached system prompt)')
    print(f'output per image is mostly thinking; assumed {args.out_low:,}-{args.out_high:,} tokens')
    print('(`review` replaces these assumptions with measured usage after the pilot)\n')
    mode = 'batch' if args.batch else 'standard'
    print(f'{"model":<20} {"mode":<9} {"low":>9} {"high":>9}')
    for m in args.models:
        lo, hi = (cost(m, inp, n * o, cache_read=n * sys_tok, batch=args.batch)
                  for o in (args.out_low, args.out_high))
        print(f'{m:<20} {mode:<9} ${lo:>8.2f} ${hi:>8.2f}')


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def out_path(out_dir, model):
    return os.path.join(out_dir, f'descriptions_{model}.jsonl')


def done_images(path):
    if not os.path.exists(path):
        return set()
    return {r['image'] for r in load(path)
            if r.get('status') == 'ok' and r.get('prompt_version') == PROMPT_VERSION}


def describe_one(client, model, it, effort, max_tokens):
    import anthropic
    rec = {'image': it['image'], 'source': it['meta']['source'], 'model': model,
           'role': role(it['ref']), 'category': it.get('category', ''),
           'prompt_version': PROMPT_VERSION, 'effort': effort, 'max_tokens': max_tokens}
    params = request_params(model, it['ref'], it['meta'], it['png'], effort, max_tokens)
    try:
        # streamed: the SDK refuses non-streaming calls with large max_tokens
        with client.messages.stream(**params) as stream:
            msg = stream.get_final_message()
            request_id = stream.request_id
    except anthropic.BadRequestError as e:
        return {**rec, 'status': 'error', 'error': f'bad request: {e.message}'}
    except anthropic.APIStatusError as e:
        return {**rec, 'status': 'error', 'error': f'{e.status_code}: {e.message}'}
    except anthropic.APIConnectionError as e:
        return {**rec, 'status': 'error', 'error': f'connection: {e}'}
    u = msg.usage
    rec.update(served_model=msg.model, request_id=request_id, stop_reason=msg.stop_reason,
               usage={'input': u.input_tokens, 'output': u.output_tokens,
                      'cache_read': u.cache_read_input_tokens or 0,
                      'cache_write': u.cache_creation_input_tokens or 0})
    if msg.stop_reason == 'refusal':
        cat = msg.stop_details.category if msg.stop_details else None
        return {**rec, 'status': 'refusal', 'error': f'refusal ({cat})'}
    if msg.stop_reason == 'max_tokens':
        return {**rec, 'status': 'error', 'error': 'hit max_tokens'}
    text = next((b.text for b in msg.content if b.type == 'text'), '')
    try:
        rec['description'] = json.loads(text)
    except json.JSONDecodeError:
        return {**rec, 'status': 'error', 'error': 'invalid JSON', 'raw': text}
    rec['status'] = 'ok'
    return rec


def cmd_run(args):
    import anthropic
    client = anthropic.Anthropic(max_retries=4)
    items = load_inputs(args)
    os.makedirs(args.out, exist_ok=True)
    for model in args.models:
        path = out_path(args.out, model)
        done = done_images(path)
        todo = [it for it in items if it['image'] not in done]
        print(f'{model}: {len(todo)} to describe ({len(done)} already done)')
        lock = threading.Lock()
        total = 0.0
        with open(path, 'a') as f, cf.ThreadPoolExecutor(args.workers) as pool:
            futs = [pool.submit(describe_one, client, model, it, args.effort, args.max_tokens)
                    for it in todo]
            for fut in cf.as_completed(futs):
                rec = fut.result()
                if 'usage' in rec:
                    total += usage_cost(model, rec['usage'])
                with lock:
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
                print(f"  {rec['image']} {rec['status']}"
                      + (f" ({rec['error']})" if rec['status'] != 'ok' else '')
                      + f'  running cost ${total:.2f}')


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------

def latest_ok(path):
    """Last successful record per image (re-runs append)."""
    out = {}
    if os.path.exists(path):
        for r in load(path):
            if r.get('status') == 'ok':
                out[r['image']] = r
    return out


def render_desc(rec):
    if rec is None:
        return '<p class="missing">no result</p>'
    d, e = rec['description'], html.escape
    parts = [f'<p class="meta">{e(d["figure_type"])} · out {rec["usage"]["output"]:,} tok · '
             f'${usage_cost(rec["model"], rec["usage"]):.3f} · prompt {e(rec["prompt_version"])}</p>',
             f'<p><b>{e(d["summary"])}</b></p>',
             f'<p class="desc">{e(d["description"])}</p>']
    if d.get('components'):  # absent in prompt v1 records
        parts.append('<details open><summary>components</summary><ul>' + ''.join(
            f'<li><b>{e(c["id"])}</b>: {e(c["type"])}' + (f' ({e(c["value"])})' if c['value'] else '')
            + '</li>' for c in d['components']) + '</ul></details>')
    if d['connections']:
        parts.append('<details open><summary>connections</summary><ul>' + ''.join(
            f'<li><b>{e(c["node"])}</b>: {e(", ".join(c["terminals"]))}</li>'
            for c in d['connections']) + '</ul></details>')
    parts.append('<details><summary>text in figure</summary><p>'
                 + e(' | '.join(d['text_in_figure'])) + '</p></details>')
    if d['uncertain']:
        parts.append('<p class="unc">uncertain: ' + e('; '.join(d['uncertain'])) + '</p>')
    return ''.join(parts)


CSS = """
body{font-family:sans-serif;margin:1em;max-width:110em}
section{border-top:2px solid #444;padding:1em 0}
.grid{display:grid;grid-template-columns:repeat(%d,1fr);gap:1em}
img{max-width:100%%;border:1px solid #ccc;background:#fff}
.desc{white-space:pre-wrap}.meta{color:#666;font-size:.85em}
.unc{color:#a50}.missing{color:#c00}.role-question{color:#c00;font-weight:bold}
details p,li{font-size:.9em}pre{white-space:pre-wrap;font-size:.8em;background:#f4f4f4;padding:.5em}
"""


def cmd_review(args):
    items = load_inputs(args)
    results = {m: latest_ok(out_path(args.out, m)) for m in args.models}
    out_html = os.path.join(args.out, 'review.html')
    rel = lambda p: os.path.relpath(p, args.out)
    e = html.escape
    body = [f'<h1>Figure description pilot — prompt {PROMPT_VERSION}</h1>']
    for it in items:
        r = role(it['ref'])
        body.append(f'<section id="{it["image"]}"><h2>{it["image"]} '
                    f'<small>{e(it.get("category", ""))} · <span class="role-{r}">{r}</span> · '
                    f'{e(it.get("reason", ""))}</small></h2>')
        body.append(f'<div class="grid"><div><a href="{e(rel(it["png"]))}">'
                    f'<img src="{e(rel(it["png"]))}"></a>'
                    f'<details><summary>prompt text</summary><pre>'
                    f'{e(user_text(it["ref"], it["meta"]))}</pre></details></div>')
        for m in args.models:
            body.append(f'<div><h3>{m}</h3>{render_desc(results[m].get(it["image"]))}</div>')
        body.append('</div></section>')
    with open(out_html, 'w') as f:
        f.write(f'<!doctype html><meta charset="utf-8"><title>pilot review</title>'
                f'<style>{CSS % (len(args.models) + 1)}</style>' + '\n'.join(body))
    print(f'wrote {out_html}')

    # Measured cost, then extrapolate to every image of the source
    all_items = load_inputs(argparse.Namespace(**{**vars(args), 'images': None}))
    full_in = sum(est_image_tokens(it['meta']) + est_text_tokens(user_text(it['ref'], it['meta']))
                  for it in all_items)
    n_all = len(all_items)
    sys_tok = est_text_tokens(SYSTEM + json.dumps(SCHEMA))
    print(f'\n{"model":<20} {"n":>3} {"pilot $":>8} {"in/img":>7} {"out/img":>8}   '
          f'full run ({n_all} images, Batch API)')
    for m in args.models:
        recs = [results[m][it['image']] for it in items if it['image'] in results[m]]
        if not recs:
            print(f'{m:<20}   0')
            continue
        n = len(recs)
        spent = sum(usage_cost(m, r['usage']) for r in recs)
        out_avg = sum(r['usage']['output'] for r in recs) / n
        in_avg = sum(r['usage']['input'] + r['usage']['cache_read'] + r['usage']['cache_write']
                     for r in recs) / n
        full = cost(m, full_in, n_all * out_avg, cache_read=n_all * sys_tok, batch=True)
        print(f'{m:<20} {n:>3} {spent:>8.2f} {in_avg:>7,.0f} {out_avg:>8,.0f}   ~${full:,.0f}')
    print('Full-run figures use measured output/image from this (deliberately hard) sample,')
    print('so they lean high; input is estimated per image from size and context length.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('estimate', 'run', 'review'):
        p = sub.add_parser(name)
        p.add_argument('--refs', required=True, help='image_refs.jsonl')
        p.add_argument('--meta', help='image_meta.jsonl (default: next to --refs)')
        p.add_argument('--png-dir', help='rendered PNGs (default: png/ next to --refs)')
        p.add_argument('--images', help='selection jsonl (e.g. pilot_images.jsonl); default: all')
        p.add_argument('--models', nargs='+', default=MODELS)
        if name == 'estimate':
            p.add_argument('--out-low', type=int, default=1500)
            p.add_argument('--out-high', type=int, default=5000)
            p.add_argument('--batch', action='store_true', help='price at Batch API rates')
        else:
            p.add_argument('--out', required=True, help='output dir')
        if name == 'run':
            p.add_argument('--effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
            p.add_argument('--workers', type=int, default=4)
            p.add_argument('--max-tokens', type=int, default=MAX_TOKENS,
                           help='output cap incl. thinking (raise to retry max_tokens errors)')
    args = ap.parse_args()
    {'estimate': cmd_estimate, 'run': cmd_run, 'review': cmd_review}[args.cmd](args)


if __name__ == '__main__':
    main()
