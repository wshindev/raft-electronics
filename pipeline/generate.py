#!/usr/bin/env python3
"""Step 6a: generate RAFT questions and cited chain-of-thought answers per chunk.

For each chunk marked `generate` in split_chunks.jsonl, Claude writes 1-4
self-contained questions answerable from that chunk, each with a reasoning answer
that quotes the chunk verbatim and a short final answer. Distractor documents are
added later, offline (step 6b). Quotes are checked against the chunk by `review`.

Usage:
    # offline cost estimate
    uv run python3 -I pipeline/generate.py estimate --data data/model [--limit 20]
    # pilot: synchronous calls on a fixed random sample
    uv run python3 -I pipeline/generate.py run --data data/model --limit 20 --out data/model/raft_pilot
    # full run on the Batch API (50% price): submit, check, collect when ended
    uv run python3 -I pipeline/generate.py submit --data data/model --split train --out data/model/raft_train
    uv run python3 -I pipeline/generate.py status --data data/model --out data/model/raft_train
    uv run python3 -I pipeline/generate.py collect --data data/model --out data/model/raft_train
    # quote check, stats, and a readable review file
    uv run python3 -I pipeline/generate.py review --data data/model --out data/model/raft_pilot

`run`, `submit`, `status` and `collect` need ANTHROPIC_API_KEY in the environment.
Batch IDs are logged in <out>/batches.jsonl; `collect` appends to the same
generated.jsonl as `run`. Errored/expired chunks are resubmitted by the next `submit`.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys

MODEL = 'claude-sonnet-5-5'
# USD per million tokens, standard rates (Batch API = 50%). Same source as describe.py.
PRICES = {'claude-sonnet-5-5': {'in': 2.00, 'out': 10.00, 'cache_read': 0.20, 'cache_write': 2.50},
          'claude-opus-5-5': {'in': 4.00, 'out': 20.00, 'cache_read': 0.20, 'cache_write': 5.00},
          'claude-fable-5-1': {'in': 10.00, 'out': 50.00, 'cache_read': 0.25, 'cache_write': 12.50}}
MAX_TOKENS = 16000
SAMPLE_SEED = 7

SYSTEM = """\
You write training questions for a model that will answer electronics questions \
with the help of retrieved documents. You get one document from an electronics \
curriculum (Tony R. Kuphaldt, Modular Electronics Learning). Write questions that \
this document answers, and answer each one by reasoning from the document.

Questions:
- Self-contained: a reader who sees only the question must know what is asked. \
Never refer to "the document", "the text", "the passage", "the author" or "the \
figure above". Give any values a calculation needs inside the question.
- Answerable from this document alone, but phrased the way a student or technician \
would ask, not by copying a sentence and blanking a word.
- Mix the kinds of question the document supports: conceptual (why/how something \
works, comparisons, consequences), numeric (a calculation the document shows or \
enables; give the numbers) and troubleshooting/application where the document \
covers it. Prefer questions that need reasoning over pure recall of a definition.
- Figures appear as placeholders like "[Figure image_2041: drawing; labels: ...]". \
You can't see the figure; the labels are only text extracted from it. Don't ask \
about a figure's content unless the document's text fully explains it, and never \
ask about the placeholder or its labels.
- Skip content that is administrative, a placeholder, or too thin to support a \
real question. Fewer good questions beat more weak ones; return an empty list if \
the document supports none.

Answers:
- reasoning: step-by-step reasoning that answers the question. Support each key \
fact with a verbatim quote from the document, written as ##begin_quote##...##end_quote##. \
Quotes must be copied exactly, character for character, from the document (keep \
LaTeX math as written), and should be short (one sentence or less). Show \
calculations with units.
- final_answer: the answer alone, short (a value with unit, a phrase, or one or two \
sentences).
- answer_type: numeric if the final answer is a number with a unit, otherwise \
conceptual.
"""

SCHEMA = {
    'type': 'object',
    'properties': {'questions': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {
            'question': {'type': 'string'},
            'answer_type': {'type': 'string', 'enum': ['numeric', 'conceptual']},
            'reasoning': {'type': 'string'},
            'final_answer': {'type': 'string'},
        },
        'required': ['question', 'answer_type', 'reasoning', 'final_answer'],
        'additionalProperties': False,
    }}},
    'required': ['questions'],
    'additionalProperties': False,
}

PROMPT_VERSION = hashlib.sha256(json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode()).hexdigest()[:12]
QUOTE_RE = re.compile(r'##begin_quote##(.*?)##end_quote##', re.S)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def n_questions(n_chars):
    return max(1, min(4, round(n_chars / 1500)))


def user_text(chunk):
    loc = chunk['locations'][0]
    where = ' > '.join(x for x in (loc.get('module_title'), loc.get('chapter'), loc.get('section')) if x)
    n = n_questions(chunk['n_chars'])
    return (f'Write up to {n} question{"s" if n > 1 else ""}.\n\n'
            f'<document where="{where}">\n{chunk["text"]}\n</document>')


def request_params(chunk, model, effort):
    return {
        'model': model,
        'max_tokens': MAX_TOKENS,
        'system': [{'type': 'text', 'text': SYSTEM, 'cache_control': {'type': 'ephemeral'}}],
        'thinking': {'type': 'adaptive'},
        'output_config': {'effort': effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        'messages': [{'role': 'user', 'content': user_text(chunk)}],
    }


def load_items(args):
    chunks = {c['id']: c for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    rows = [r for r in load(os.path.join(args.data, 'split_chunks.jsonl')) if r['generate']]
    if args.split:
        rows = [r for r in rows if r['split'] == args.split]
    if args.limit:
        rows = random.Random(SAMPLE_SEED).sample(rows, min(args.limit, len(rows)))
    return [(r, chunks[r['id']]) for r in rows]


def cost(model, inp, out, cache_read=0, cache_write=0, batch=False):
    p = PRICES[model]
    usd = (inp * p['in'] + out * p['out'] + cache_read * p['cache_read'] + cache_write * p['cache_write']) / 1e6
    return usd / 2 if batch else usd


def usage_cost(model, u, batch=False):
    return cost(model, u['input'], u['output'], u['cache_read'], u['cache_write'], batch=batch)


def cmd_estimate(args):
    items = load_items(args)
    sys_tok = len(SYSTEM + json.dumps(SCHEMA)) / 3.0
    inp = sum(len(user_text(c)) / 3.0 for _, c in items)
    nq = sum(n_questions(c['n_chars']) for _, c in items)
    n = len(items)
    print(f'{n} chunks, up to {nq} questions, prompt {PROMPT_VERSION}')
    print(f'est. input {inp / n:,.0f} tokens/chunk (+ ~{sys_tok:,.0f} cached system prompt)')
    print(f'output assumed {args.out_low:,}-{args.out_high:,} tokens/chunk (thinking + JSON)')
    for batch in (False, True):
        lo, hi = (cost(args.model, inp, n * o, cache_read=n * sys_tok, batch=batch)
                  for o in (args.out_low, args.out_high))
        print(f'{args.model} {"batch" if batch else "standard":<9} ${lo:,.2f} - ${hi:,.2f}')


def out_path(out_dir):
    return os.path.join(out_dir, 'generated.jsonl')


def base_rec(row, chunk, model, effort):
    return {'chunk_id': chunk['id'], 'source': chunk['source'], 'split': row['split'],
            'chunk_type': chunk['type'], 'model': model, 'effort': effort,
            'prompt_version': PROMPT_VERSION}


def gen_one(client, row, chunk, model, effort):
    import anthropic
    rec = base_rec(row, chunk, model, effort)
    try:
        with client.messages.stream(**request_params(chunk, model, effort)) as stream:
            msg = stream.get_final_message()
            rec['request_id'] = stream.request_id
    except anthropic.APIStatusError as e:
        return {**rec, 'status': 'error', 'error': f'{e.status_code}: {e.message}'}
    except anthropic.APIConnectionError as e:
        return {**rec, 'status': 'error', 'error': f'connection: {e}'}
    return parse_message(rec, msg)


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
        rec['questions'] = json.loads(text)['questions']
    except (json.JSONDecodeError, KeyError):
        return {**rec, 'status': 'error', 'error': 'invalid JSON', 'raw': text}
    rec['status'] = 'ok'
    return rec


def cmd_run(args):
    import anthropic
    client = anthropic.Anthropic(max_retries=4)
    items = load_items(args)
    os.makedirs(args.out, exist_ok=True)
    path = out_path(args.out)
    done = set()
    if os.path.exists(path):
        done = {r['chunk_id'] for r in load(path)
                if r['status'] == 'ok' and r['prompt_version'] == PROMPT_VERSION}
    todo = [(r, c) for r, c in items if c['id'] not in done]
    print(f'{len(todo)} chunks to do ({len(done)} done), prompt {PROMPT_VERSION}')
    total = 0.0
    with open(path, 'a') as f, cf.ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(gen_one, client, r, c, args.model, args.effort) for r, c in todo]
        for fut in cf.as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + '\n')  # save before anything else can fail
            f.flush()
            if 'usage' in rec:
                total += usage_cost(args.model, rec['usage'])
            print(f"  {rec['chunk_id']} {rec['status']} {len(rec.get('questions', []))} q"
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


def cmd_submit(args, client=None):
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)
    items = load_items(args)
    os.makedirs(args.out, exist_ok=True)
    done = set()
    if os.path.exists(out_path(args.out)):
        done = {r['chunk_id'] for r in load(out_path(args.out))
                if r['status'] == 'ok' and r['prompt_version'] == PROMPT_VERSION}
    pending = {cid for b in load_batches(args.out).values() if not b.get('collected')
               for cid in b['chunk_ids']}
    todo = [(r, c) for r, c in items if c['id'] not in done | pending]
    print(f'{len(todo)} chunks to submit ({len(done)} done, {len(pending)} in uncollected batches), '
          f'{args.model}, effort {args.effort}, prompt {PROMPT_VERSION}')
    if not todo:
        return
    for _, c in todo:
        assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}', c['id']), c['id']
    requests = [{'custom_id': c['id'], 'params': request_params(c, args.model, args.effort)}
                for _, c in todo]
    size = len(json.dumps(requests))
    assert len(requests) <= 100_000 and size < 200e6, (len(requests), size)  # API limits: 100k / 256 MB
    batch = client.messages.batches.create(requests=requests)
    log_batch(args.out, {'batch_id': batch.id, 'model': args.model, 'effort': args.effort,
                         'prompt_version': PROMPT_VERSION, 'split': args.split,
                         'chunk_ids': [c['id'] for _, c in todo]})
    print(f'submitted {batch.id}: {len(requests)} requests, {size / 1e6:.1f} MB')


def cmd_status(args, client=None):
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)
    for bid, b in load_batches(args.out).items():
        if b.get('collected'):
            print(f'{bid} collected ({len(b["chunk_ids"])} chunks)')
            continue
        info = client.messages.batches.retrieve(bid)
        n = info.request_counts
        print(f'{bid} {info.processing_status}: processing {n.processing}, succeeded {n.succeeded}, '
              f'errored {n.errored}, canceled {n.canceled}, expired {n.expired}')


def cmd_collect(args, client=None):
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)
    chunks = {c['id']: c for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    rows = {r['id']: r for r in load(os.path.join(args.data, 'split_chunks.jsonl'))}
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
                rec = base_rec(rows[res.custom_id], chunks[res.custom_id], b['model'], b['effort'])
                rec.update(batch_id=bid, batch=True)
                rec['prompt_version'] = b['prompt_version']
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


def check_quotes(q, chunk_text):
    """Fraction of quotes found verbatim in the chunk (whitespace-normalized)."""
    norm = lambda s: re.sub(r'\s+', ' ', s).strip()
    text = norm(chunk_text)
    quotes = [norm(x) for x in QUOTE_RE.findall(q['reasoning'])]
    found = sum(1 for x in quotes if x and x in text)
    return len(quotes), found


def cmd_review(args):
    chunks = {c['id']: c for c in load(os.path.join(args.data, 'chunks_clean.jsonl'))}
    latest = {}
    for r in load(out_path(args.out)):
        if r['status'] == 'ok':
            latest[r['chunk_id']] = r
    recs = list(latest.values())
    nq = nquotes = nfound = no_quote_q = bad_q = 0
    md = []
    for r in recs:
        c = chunks[r['chunk_id']]
        loc = c['locations'][0]
        md += [f'## {c["id"]} ({c["type"]}, {r["split"]}) — {loc.get("module_title")} > {loc.get("section") or loc.get("chapter")}',
               '', '<details><summary>document</summary>', '', c['text'], '', '</details>', '']
        for q in r['questions']:
            total, found = check_quotes(q, c['text'])
            nq += 1
            nquotes += total
            nfound += found
            no_quote_q += total == 0
            bad_q += found < total
            flag = '' if total and found == total else f'  **⚠ quotes {found}/{total} verbatim**'
            md += [f'**Q ({q["answer_type"]}):** {q["question"]}{flag}', '',
                   f'**Final:** {q["final_answer"]}', '', f'*Reasoning:* {q["reasoning"]}', '']
        md.append('---')
    with open(os.path.join(args.out, 'review.md'), 'w') as f:
        f.write(f'# RAFT generation pilot — prompt {PROMPT_VERSION}\n\n' + '\n'.join(md))
    usage = [r['usage'] for r in recs]
    spent = sum(usage_cost(r['model'], r['usage'], r.get('batch', False)) for r in recs)
    out_avg = sum(u['output'] for u in usage) / max(len(usage), 1)
    types = {}
    for r in recs:
        for q in r['questions']:
            types[q['answer_type']] = types.get(q['answer_type'], 0) + 1
    print(f'{len(recs)} chunks, {nq} questions {types}; chunks with no questions: '
          f'{sum(1 for r in recs if not r["questions"])}')
    print(f'quotes: {nfound}/{nquotes} verbatim; questions without quotes {no_quote_q}, '
          f'with a non-verbatim quote {bad_q}')
    print(f'cost ${spent:.2f}, output {out_avg:,.0f} tokens/chunk')
    # extrapolate to every generate chunk, Batch API
    all_rows = [r for r in load(os.path.join(args.data, 'split_chunks.jsonl')) if r['generate']]
    full_in = sum(len(user_text(chunks[r['id']])) / 3.0 for r in all_rows)
    m = recs[0]['model'] if recs else MODEL
    sys_tok = len(SYSTEM + json.dumps(SCHEMA)) / 3.0
    full = cost(m, full_in, len(all_rows) * out_avg, cache_read=len(all_rows) * sys_tok, batch=True)
    print(f'full run: {len(all_rows)} chunks on Batch API ≈ ${full:,.0f}')
    print(f'wrote {os.path.join(args.out, "review.md")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    cmds = {'estimate': cmd_estimate, 'run': cmd_run, 'submit': cmd_submit, 'status': cmd_status,
            'collect': cmd_collect, 'review': cmd_review}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='data/<source> folder (steps 04-05 outputs)')
        if name in ('estimate', 'run', 'submit'):
            p.add_argument('--limit', type=int, help='random sample of N chunks (fixed seed)')
            p.add_argument('--split', choices=['train', 'heldout'])
            p.add_argument('--model', default=MODEL, choices=sorted(PRICES))
        if name == 'estimate':
            p.add_argument('--out-low', type=int, default=1500)
            p.add_argument('--out-high', type=int, default=4000)
        else:
            p.add_argument('--out', required=True, help='output dir')
        if name in ('run', 'submit'):
            p.add_argument('--effort', default='medium', choices=['low', 'medium', 'high'])
        if name == 'run':
            p.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    cmds[args.cmd](args)


if __name__ == '__main__':
    main()
