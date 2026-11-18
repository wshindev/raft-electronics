#!/usr/bin/env python3
"""Step 07: run eval prompt sets against a llama-server model. Stdlib only.

One model config per server (e.g. E4B base, E4B + Sonnet LoRA), each at the same
quantization. Requests use /v1/chat/completions with the server's chat template (start
llama-server with --jinja), thinking off via chat_template_kwargs (--think: on), greedy decoding
(temperature 0) and a fixed max output length (4,096: at 1,024 the base model's answers were cut). Raw outputs are stored; scoring is a
separate offline step.

  check  render one prompt through /apply-template with thinking off and on, and show the
         end of the prompt: E4B must end at "<|turn>model\\n", 12B/31B with the empty thought
         block "<|channel>thought\\n<channel|>". Then generate one answer.
  run    every item of each --inputs file; items already in the output are skipped, so a
         stopped run resumes. Any server error stops the run.

Usage:
    uv run python3 -I pipeline/eval_run.py check --url http://HOST:8080 --input data/model/eval_sets/fable_top5.jsonl
    uv run python3 -I pipeline/eval_run.py run --url http://HOST:8080 --config e4b_base \\
        --inputs data/model/eval_sets/*.jsonl --out data/model/eval_runs [--limit 20]

If the server needs a key, set LLAMA_API_KEY in the environment (sent as a Bearer token).

--lora-scale S sends the server's single --lora adapter at strength S with every request
(llama-server's per-request "lora" field; 0 = base, 1 = the full adapter). Used for the
adapter-strength test; give each strength its own --config.

--think turns Gemma's thinking mode on (chat_template_kwargs enable_thinking: true); default off,
as in training. The server must split the thinking from the answer (llama-server's default
--reasoning-format), so the thinking lands in `reasoning` and only the visible answer in
`output`, which is all that scoring reads. completion_tokens then counts thinking + answer.
Used for the instruct-model-with-thinking baseline; give it its own --config and
a larger --max-tokens.

--append-file F appends the text of F to the end of each item's user message (after a blank
line), e.g. an extra instruction for Test A (pipeline/prompts/test_a_brief.txt).
The text is recorded in meta.json; give each instruction its own --config.

Writes to <out>/<config>/:
    <input file name>   {id, set, condition, output, reasoning, finish_reason, prompt_tokens,
                        completion_tokens, seconds} per item
    meta.json           config, server models and settings (/props), request parameters
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get('LLAMA_API_KEY', '')
THOUGHT = '<|channel>thought\n<channel|>'


def headers():
    h = {'Content-Type': 'application/json'}
    if API_KEY:
        h['Authorization'] = f'Bearer {API_KEY}'
    return h


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def request(url, path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip('/') + path, data=data, headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{path}: server error {e.code}: {e.read()[:500]!r}') from None


def server_info(url, timeout):
    info = {}
    for path in ('/v1/models', '/props', '/lora-adapters'):
        try:
            info[path] = request(url, path, timeout=timeout)
        except (RuntimeError, urllib.error.URLError, ValueError) as e:
            info[path] = f'unavailable: {e}'
    models = info['/v1/models']
    props = info['/props'] if isinstance(info['/props'], dict) else {}
    gen = props.get('default_generation_settings', {})
    return {'models': [m.get('id') for m in models.get('data', [])] if isinstance(models, dict) else models,
            'lora_adapters': info['/lora-adapters'],  # /v1/models doesn't show a --lora adapter
            'n_ctx_per_slot': gen.get('n_ctx'), 'total_slots': props.get('total_slots'),
            'chat_template_chars': len(props.get('chat_template', '') or ''),
            'build_info': props.get('build_info')}


def messages(args, item):
    msgs = item['messages']
    if not args.append:
        return msgs
    *head, last = msgs
    if last['role'] != 'user':
        sys.exit(f'{item["id"]}: last message is not the user turn; can\'t append')
    return head + [{**last, 'content': last['content'].rstrip() + '\n\n' + args.append}]


def payload(args, msgs):
    p = {'messages': msgs, 'temperature': 0.0, 'top_k': 1, 'max_tokens': args.max_tokens, 'seed': 0,
         'chat_template_kwargs': {'enable_thinking': args.think}, 'cache_prompt': True}
    if args.lora_scale is not None:
        p['lora'] = [{'id': 0, 'scale': args.lora_scale}]
    return p


def params(args):
    """Request settings as recorded in meta.json (the appended text included)."""
    p = {k: v for k, v in payload(args, []).items() if k != 'messages'}
    if args.append:
        p['append'] = args.append
    return p


def check_lora(args, info):
    if args.lora_scale is None:
        return
    ad = info['lora_adapters']
    if not isinstance(ad, list) or len(ad) != 1 or ad[0].get('id') != 0:
        sys.exit(f'--lora-scale needs exactly one adapter loaded with --lora (id 0); server reports {ad}')


def generate(args, item):
    t0 = time.time()
    res = request(args.url, '/v1/chat/completions', payload(args, messages(args, item)), args.timeout)
    ch, usage = res['choices'][0], res.get('usage', {})
    return {'id': item['id'], 'set': item.get('set'), 'condition': item.get('condition'),
            'output': ch['message'].get('content') or '', 'reasoning': ch['message'].get('reasoning_content'),
            'finish_reason': ch.get('finish_reason'), 'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'), 'seconds': round(time.time() - t0, 2)}


def cmd_check(args):
    info = server_info(args.url, args.timeout)
    print('server:', json.dumps(info))
    check_lora(args, info)
    item = load(args.input)[0]
    renders = {}
    for think in (False, True):
        renders[think] = request(args.url, '/apply-template',
                                 {'messages': messages(args, item), 'chat_template_kwargs': {'enable_thinking': think}},
                                 args.timeout)['prompt']
    off = renders[False]
    print(f'--- end of the prompt, thinking off ({item["id"]}) ---\n{off[-300:]}⟂')
    if off == renders[True]:
        print('WARNING: thinking on and off render the same prompt; chat_template_kwargs is not reaching the '
              'template (start llama-server with --jinja).')
    if off.endswith(THOUGHT):
        print('prompt ends with the empty thought block (expected for 12B/31B, not for E4B)')
    elif off.endswith('<|turn>model\n'):
        print('prompt ends at "<|turn>model\\n", no thought block (expected for E4B, not for 12B/31B)')
    else:
        print('WARNING: the prompt does not end with the Gemma 4 generation prompt')
    tok = request(args.url, '/tokenize', {'content': off, 'add_special': False}, args.timeout)['tokens']
    print(f'prompt tokens: {len(tok)} (+ max output {args.max_tokens}); server context per slot: '
          f'{info["n_ctx_per_slot"] or "unknown"}; LoRA adapters: {info["lora_adapters"]}')
    out = generate(args, item)
    print(f'--- output ({out["completion_tokens"]} tokens, finish {out["finish_reason"]}, {out["seconds"]} s) ---')
    print(out['output'])
    if out['reasoning']:
        print(f'--- reasoning_content ({len(out["reasoning"])} chars; should be empty with thinking off) ---\n'
              f'{out["reasoning"]}')
    for w in leak_warnings(args, out):
        print('WARNING:', w)


def leak_warnings(args, r):
    """With --think: the thinking must come back separately, not inside the answer text."""
    if not args.think:
        return []
    w = []
    if not r['reasoning'] and r['finish_reason'] != 'length':
        w.append(f'{r["id"]}: no reasoning_content with thinking on (server not splitting it, or the model '
                 f'skipped thinking)')
    if '<|channel>' in r['output'] or '<channel|>' in r['output']:
        w.append(f'{r["id"]}: thought-channel tags in the answer text; the server is not separating the '
                 f'thinking (check --reasoning-format)')
    return w


def cmd_run(args):
    out_dir = os.path.join(args.out, args.config)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, 'meta.json')
    info = server_info(args.url, args.timeout)
    if not isinstance(info['models'], list):  # don't record an unreachable server in meta.json
        sys.exit(f'server not reachable: {info["models"]}\n(--url needs http://; a 401 means LLAMA_API_KEY is '
                 f'missing or wrong)')
    check_lora(args, info)
    params_ = params(args)
    if os.path.exists(meta_path):  # a resumed run must use the same model and settings
        old = json.load(open(meta_path))
        new = {'models': info['models'], 'lora_adapters': info['lora_adapters'], 'params': params_}
        diff = [f'  {k}: earlier {old.get(k)} | now {v}' for k, v in new.items() if old.get(k) != v]
        if diff:
            sys.exit(f'{meta_path}: differs from the earlier run in this folder:\n' + '\n'.join(diff) +
                     '\nUse another --config (or delete the folder if the earlier run produced nothing).')
    with open(meta_path, 'w') as f:
        json.dump({'config': args.config, **info, 'params': params_, 'inputs': args.inputs}, f, indent=1)
    for path in args.inputs:
        out = os.path.join(out_dir, os.path.basename(path))
        done = {r['id'] for r in load(out)} if os.path.exists(out) else set()
        items = load(path)[:args.limit or None]
        todo = [it for it in items if it['id'] not in done]
        print(f'{os.path.basename(path)}: {len(items) - len(todo)} done, {len(todo)} to go')
        length, leaks = 0, 0
        with open(out, 'a') as f, ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(generate, args, it): it for it in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    r = fut.result()
                except (RuntimeError, urllib.error.URLError, TimeoutError) as e:
                    for other in futs:
                        other.cancel()
                    sys.exit(f'{futs[fut]["id"]}: {e}\nRerun the same command to resume.')
                length += r['finish_reason'] == 'length'
                leaks += bool(leak_warnings(args, r))
                f.write(json.dumps(r) + '\n')
                f.flush()
                print(f'\r  {n}/{len(todo)}', end='', flush=True)
        print(f'\n  wrote {out}' + (f'; {length} outputs hit max tokens' if length else '') +
              (f'; {leaks} outputs without separated thinking (see check)' if leaks else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--url', required=True, help='llama-server base URL, e.g. http://localhost:8080')
        p.add_argument('--max-tokens', type=int, default=4096,
                       help='max output tokens (training answers: max 642; the base model writes long answers)')
        p.add_argument('--timeout', type=float, default=900)
        p.add_argument('--lora-scale', type=float, help='strength of the loaded --lora adapter per request')
        p.add_argument('--append-file', help='text file appended to the end of each user message')
        p.add_argument('--think', action='store_true', help='thinking mode on (default off, as in training)')
        if name == 'check':
            p.add_argument('--input', required=True, help='eval_sets.py output file; the first item is used')
        else:
            p.add_argument('--config', required=True, help='names the output folder, e.g. e4b_base, e4b_sonnet')
            p.add_argument('--inputs', nargs='+', required=True, help='eval_sets.py output files')
            p.add_argument('--out', required=True)
            p.add_argument('--workers', type=int, default=4, help='concurrent requests (match the server --parallel)')
            p.add_argument('--limit', type=int, default=0, help='first N items per file (pilot)')
    args = ap.parse_args()
    args.append = ''
    if args.append_file:
        with open(args.append_file) as f:
            args.append = f.read().strip()
        if not args.append:
            sys.exit(f'{args.append_file} is empty')
    {'check': cmd_check, 'run': cmd_run}[args.cmd](args)


if __name__ == '__main__':
    main()
