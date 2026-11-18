#!/usr/bin/env python3
"""Step 07: chat-template check for fine-tuning. CPU, tokenizer only.

Runs in the train/ venv (transformers). For each model's tokenizer folder:
  1. Renders one SFT example with thinking off, as the full conversation (training) and
     as the prompt alone with the generation prompt (inference), and checks that the
     prompt is a token-level prefix of the full conversation. That's what makes
     "loss on the assistant span only" well defined: the masked span = the prompt tokens.
  2. Shows the boundary (end of prompt, start of the assistant span) and whether an
     empty thought block appears there (12B/31B emit one with thinking off, E4B doesn't).
  3. Renders with thinking on as well, to confirm the thinking switch reaches the template.
  4. Counts exact token lengths of every example in the given SFT sets (full and
     assistant span) and how many exceed --max-len.

Usage:
    train/.venv/bin/python -I pipeline/template_check.py \\
        --tokenizer train/tokenizers/gemma-4-E4B-it train/tokenizers/gemma-4-12B-it train/tokenizers/gemma-4-31B-it \\
        --sft data/model/sft_sonnet_hybrid data/model/sft_opus_hybrid --out data/model/template_check

Writes to --out: report.md, lengths_<tokenizer>.jsonl ({id, set, split, total, assistant} per example)
"""
import argparse
import hashlib
import json
import os

from transformers import AutoTokenizer

CTX = 300  # characters shown on each side of the prompt/assistant boundary


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def render(tok, messages, **kw):
    return tok.apply_chat_template(messages, tokenize=False, **kw)


def ids(tok, text):
    return tok(text, add_special_tokens=False)['input_ids']


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def check_one(tok, ex, think):
    """(full text, prompt text, full ids, prompt ids, is_prefix) for one example."""
    msgs = ex['messages']
    full = render(tok, msgs, enable_thinking=think)
    prompt = render(tok, msgs[:-1], add_generation_prompt=True, enable_thinking=think)
    f, p = ids(tok, full), ids(tok, prompt)
    return full, prompt, f, p, f[:len(p)] == p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokenizer', nargs='+', required=True, help='model folders with tokenizer files')
    ap.add_argument('--sft', nargs='+', required=True, help='assemble.py output folders (train.jsonl, dev.jsonl)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-len', type=int, default=10240)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for d in args.sft:
        for split in ('train', 'dev'):
            for r in load(os.path.join(d, f'{split}.jsonl')):
                rows.append((os.path.basename(os.path.normpath(d)), split, r))
    sample = min((r for r in rows if r[2]['oracle_present']), key=lambda r: len(r[2]['messages'][1]['content']))[2]

    lines = [f'# Chat-template check ({len(rows)} examples from {", ".join(args.sft)})', '']
    seen = {}
    for tdir in args.tokenizer:
        name = os.path.basename(os.path.normpath(tdir))
        tok = AutoTokenizer.from_pretrained(tdir, local_files_only=True, trust_remote_code=False)
        tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
        sig = hashlib.sha1((tmpl or '').encode() + json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()[:12]
        lines += [f'## {name}', '', f'- tokenizer class `{type(tok).__name__}`, vocab {len(tok)}, '
                  f'template+vocab signature `{sig}`', f'- bos `{tok.bos_token}`, eos `{tok.eos_token}`, pad `{tok.pad_token}`']

        full, prompt, f, p, ok = check_one(tok, sample, think=False)
        full_on, prompt_on, *_ = check_one(tok, sample, think=True)
        tail = full[len(prompt):] if full.startswith(prompt) else None
        lines += [f'- prompt is a **token** prefix of the full conversation: **{ok}** '
                  f'(text prefix: {full.startswith(prompt)}); prompt {len(p)} tokens, full {len(f)}',
                  f'- thinking switch changes the render: prompt {prompt_on != prompt}, full {full_on != full}',
                  f'- `<|channel>` in prompt end / assistant span: {"<|channel>" in prompt[-CTX:]} / '
                  f'{tail is not None and "<|channel>" in tail}', '',
                  'Boundary, thinking off (prompt end ⟂ assistant span):', '```',
                  prompt[-CTX:] + '⟂' + (tail[:CTX] if tail is not None else '<full render does not start with the prompt>'),
                  '```', 'End of the full render:', '```', full[-CTX:], '```', '',
                  'Generation prompt, thinking on (end):', '```', prompt_on[-CTX:], '```',
                  'Beginning of the full render (system + first document):', '```', full[:CTX * 2], '```', '']

        if sig in seen:
            lines += [f'Same tokenizer and template as `{seen[sig]}`: lengths not recounted.', '']
            continue
        seen[sig] = name
        lens, bad_prefix = [], 0
        for sname, split, r in rows:
            _, _, f, p, ok = check_one(tok, r, think=False)
            bad_prefix += not ok
            lens.append({'id': r['id'], 'set': sname, 'split': split, 'total': len(f), 'assistant': len(f) - len(p)})
        with open(os.path.join(args.out, f'lengths_{name}.jsonl'), 'w') as fh:
            for x in lens:
                fh.write(json.dumps(x) + '\n')
        lines += [f'Token lengths, thinking off ({len(lens)} examples; prompt not a token prefix: {bad_prefix}):', '',
                  '| set | n | total median | p90 | p99 | max | > max-len | assistant median | max |', '|---' * 9 + '|']
        for sname in sorted({x['set'] for x in lens}):
            t = [x['total'] for x in lens if x['set'] == sname]
            a = [x['assistant'] for x in lens if x['set'] == sname]
            over = sum(v > args.max_len for v in t)
            lines.append(f'| {sname} | {len(t)} | {pct(t, .5):,} | {pct(t, .9):,} | {pct(t, .99):,} | {max(t):,} | '
                         f'{over} | {pct(a, .5):,} | {max(a):,} |')
        lines += ['', f'Chars per token (all examples): '
                  f'{sum(len(r[2]["messages"][1]["content"]) for r in rows) / sum(x["total"] for x in lens):.2f} '
                  '(user content chars / total tokens, approximate)', '']

    text = '\n'.join(lines)
    with open(os.path.join(args.out, 'report.md'), 'w') as fh:
        fh.write(text)
    print(text)
    print(f'wrote {args.out}/report.md')


if __name__ == '__main__':
    main()
