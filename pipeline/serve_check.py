#!/usr/bin/env python3
"""Step 07: does the served adapter (llama-server, LoRA GGUF on the Q8_0 base) behave like the
trained one (transformers + PEFT, bf16)? Runs in the train/ venv on the GPU.

The eval scores all come from llama-server with the adapter converted to GGUF. This script loads
the original adapter (runs/<run>/adapter/) on the bf16 base, the way training produced it, and
for each item:
  1. generates greedily from the same prompt (thinking off, as in training and eval_run.py) and
     shows that output next to the served one;
  2. scores the served outputs token by token (teacher forcing): mean negative log-likelihood
     per token of the served adapter's output and of the served base's output, under the trained
     adapter and under the bf16 base (adapter disabled).
Greedy outputs can drift apart from small numeric differences (Q8_0 vs bf16) even when the
conversion is right, so (2) is the main test. If the served adapter is faithful, its outputs are
about as likely under the trained adapter as the served base's outputs are under the bf16 base
(both differ only by quantization), and clearly more likely under the adapter than under the base.

Usage (stop llama-server first; E4B bf16 needs ~16 GB):
    train/.venv/bin/python -I pipeline/serve_check.py --model <models>/gemma-4-E4B-it \\
        --adapter data/model/runs/e4b_sonnet/adapter --input data/model/eval_sets/exam_nodocs.jsonl \\
        --served data/model/eval_runs/e4b_sonnet/exam_nodocs.jsonl \\
        --served-base data/model/eval_runs/e4b_base/exam_nodocs.jsonl \\
        --ids xq_1387 xq_1664 ... --out data/model/serve_check/e4b_sonnet

Writes to --out: items.jsonl (per item: both outputs, NLLs) and report.md.
"""
import argparse
import json
import os
import statistics
import sys
import time

TURN_END = '<turn|>'


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def final(text):
    """The text after the last <ANSWER> marker (colon optional), else the last 200 chars."""
    i = text.rfind('<ANSWER>')
    return (text[i + len('<ANSWER>'):].lstrip(':').strip() if i >= 0 else text[-200:]).replace('\n', ' ')[:300]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='local bf16 base model folder (the one training used)')
    ap.add_argument('--adapter', required=True, help='trained adapter folder (runs/<run>/adapter)')
    ap.add_argument('--input', required=True, help='eval_sets prompt file')
    ap.add_argument('--served', required=True, help='eval_run.py output of the adapter config for that file')
    ap.add_argument('--served-base', help='eval_run.py output of the base config for that file')
    ap.add_argument('--ids', nargs='+', help='item ids (default: the first --limit items)')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--max-new-tokens', type=int, default=4096, help='same as eval_run.py')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    args.model = os.path.expanduser(args.model)

    items = {r['id']: r for r in load(args.input)}
    served = {r['id']: r for r in load(args.served)}
    served_base = {r['id']: r for r in load(args.served_base)} if args.served_base else {}
    ids = args.ids or list(items)[:args.limit]
    missing = [i for i in ids if i not in items or i not in served or (served_base and i not in served_base)]
    if missing:
        sys.exit(f'not in the input/served files: {missing}')
    os.makedirs(args.out, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda',
                                                       local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    turn_end_id = tok.convert_tokens_to_ids(TURN_END)
    eos = sorted({turn_end_id, *([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else tok.eos_token_id or [])})
    print(f'loaded {args.model} + {args.adapter}; stop ids {eos}')

    def prompt_ids(item):
        text = tok.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return tok(text, add_special_tokens=False)['input_ids']

    @torch.no_grad()
    def nll(p, output, adapter):
        """Mean NLL per token of output + <turn|> after prompt ids p."""
        o = tok(output + TURN_END, add_special_tokens=False)['input_ids']
        x = torch.tensor([p + o], device='cuda')
        if adapter:
            logits = model(input_ids=x).logits
        else:
            with model.disable_adapter():
                logits = model(input_ids=x).logits
        lp = torch.log_softmax(logits[0, len(p) - 1:-1].float(), dim=-1)
        tgt = x[0, len(p):]
        return -lp.gather(1, tgt[:, None]).mean().item(), len(o)

    rows = []
    for n, i in enumerate(ids, 1):
        t0 = time.time()
        p = prompt_ids(items[i])
        with torch.no_grad():
            g = model.generate(input_ids=torch.tensor([p], device='cuda'), do_sample=False,
                               max_new_tokens=args.max_new_tokens, eos_token_id=eos)
        hf = tok.decode(g[0, len(p):], skip_special_tokens=True).strip()
        sv = served[i]['output'].strip()
        row = {'id': i, 'hf_output': hf, 'served_output': sv, 'identical': hf == sv,
               'common_prefix_chars': common_prefix(hf, sv), 'hf_final': final(hf), 'served_final': final(sv)}
        row['nll_served_ft_under_ft'], row['tokens_served_ft'] = nll(p, sv, True)
        row['nll_served_ft_under_base'], _ = nll(p, sv, False)
        if served_base:
            sb = served_base[i]['output'].strip()
            row['nll_served_base_under_base'], row['tokens_served_base'] = nll(p, sb, False)
            row['nll_served_base_under_ft'], _ = nll(p, sb, True)
        rows.append(row)
        print(f'[{n}/{len(ids)}] {i}: identical {row["identical"]}, common prefix {row["common_prefix_chars"]} of '
              f'{len(sv)} chars, NLL served-FT under FT {row["nll_served_ft_under_ft"]:.3f} / under base '
              f'{row["nll_served_ft_under_base"]:.3f} ({time.time() - t0:.0f}s)')

    with open(os.path.join(args.out, 'items.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    med = lambda k: statistics.median(r[k] for r in rows if k in r)
    lines = [f'# Served vs trained adapter', '',
             f'Trained: `{args.adapter}` on `{args.model}` (bf16, transformers + PEFT). Served: `{args.served}`. '
             f'{len(rows)} items from `{args.input}`, greedy, thinking off.', '',
             '## Teacher-forced NLL per token (median over items; lower = more likely)', '',
             '| output scored | under trained adapter | under bf16 base |', '|---|---|---|',
             f'| served adapter output | {med("nll_served_ft_under_ft"):.3f} | {med("nll_served_ft_under_base"):.3f} |']
    if served_base:
        lines.append(f'| served base output | {med("nll_served_base_under_ft"):.3f} | '
                     f'{med("nll_served_base_under_base"):.3f} |')
    lines += ['', 'Faithful conversion: served-adapter output under the trained adapter is low, about like '
              'served-base output under the bf16 base (both differ only by Q8_0 vs bf16), and well below the '
              'served-adapter output under the base.', '',
              f'Greedy outputs identical: {sum(r["identical"] for r in rows)} of {len(rows)}. '
              f'Same final answer text: {sum(r["hf_final"] == r["served_final"] for r in rows)} of {len(rows)}.', '',
              '## Per item', '',
              '| id | identical | common prefix / served chars | NLL served-FT: FT / base | final, trained | final, served |',
              '|---|---|---|---|---|---|']
    for r in rows:
        cell = lambda s: s.replace('|', '\\|')[:120]
        lines.append(f'| {r["id"]} | {r["identical"]} | {r["common_prefix_chars"]} / {len(r["served_output"])} | '
                     f'{r["nll_served_ft_under_ft"]:.3f} / {r["nll_served_ft_under_base"]:.3f} | '
                     f'{cell(r["hf_final"])} | {cell(r["served_final"])} |')
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.out, 'report.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
