#!/usr/bin/env python3
"""Step 07: LoRA / QLoRA fine-tuning with Unsloth. Runs in the train/ venv.

Training text per example (template check): the generation prompt rendered with
thinking off, then the assistant content, then the end-of-turn token `<turn|>`:
    apply_chat_template(messages[:-1], add_generation_prompt=True, enable_thinking=False)
    + messages[-1].content + "<turn|>"
For 12B/31B the prompt ends with the empty thought block the model sees at inference
(`<|channel>thought\\n<channel|>`), which a plain apply_chat_template(messages) would drop.
Loss is on the assistant span only (labels -100 on the prompt tokens). Every example is
checked: the prompt's tokens are an exact prefix of the training tokens, the span ends
with `<turn|>` (one of the model's EOS ids), and the span's tokens equal the end of the
full-conversation render. Examples longer than --max-len are dropped and listed, never cut;
so are examples with an empty final answer (one in sft_sonnet_hybrid, its reasoning is cut off).

Modes:
  --dry-run       tokenizer only (CPU, no Unsloth/torch import): build + check every
                  example, print stats and one decoded boundary, write nothing.
  --pilot-steps N train N optimizer steps on the N x grad-accum longest train examples,
                  no eval, no adapter saved; logs peak GPU memory and tokens/s.
  (default)       full run: 1 epoch, dev loss every --eval-steps, adapter saved to --out.

Usage:
    train/.venv/bin/python -I pipeline/train.py --dry-run --model train/tokenizers/gemma-4-E4B-it \\
        --sft data/model/sft_sonnet_hybrid
    train/.venv/bin/python -I pipeline/train.py --model <models>/gemma-4-E4B-it \\
        --sft data/model/sft_sonnet_hybrid --out data/model/runs/e4b_sonnet_pilot --pilot-steps 20
    train/.venv/bin/python -I pipeline/train.py --model <models>/gemma-4-E4B-it \\
        --sft data/model/sft_sonnet_hybrid --out data/model/runs/e4b_sonnet

Writes to --out: config.json (all settings, data/template hashes, dropped ids), log.jsonl
(one line per logging step: loss, lr, grad norm, tokens seen, tokens/s, peak memory),
summary.json, and for full runs adapter/ (LoRA weights + tokenizer) and checkpoints/.
"""
import argparse
import hashlib
import json
import os
import sys
import time

TURN_END = '<turn|>'


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()[:16]


def build(tok, rows, max_len):
    """Tokenize and check every example. Returns (examples, dropped ids)."""
    turn_end_id = tok.convert_tokens_to_ids(TURN_END)
    if tok.convert_ids_to_tokens(turn_end_id) != TURN_END:
        sys.exit(f'{TURN_END} is not a single token in this tokenizer')
    out, dropped = [], []
    for r in rows:
        msgs = r['messages']
        if msgs[-1]['role'] != 'assistant':
            sys.exit(f'{r["id"]}: last message is not the assistant turn')
        content = msgs[-1]['content']
        if not content.rpartition('<ANSWER>:')[2].strip():
            dropped.append({'id': r['id'], 'reason': 'empty final answer'})
            continue
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        p = tok(prompt, add_special_tokens=False)['input_ids']
        x = tok(prompt + content + TURN_END, add_special_tokens=False)['input_ids']
        span = x[len(p):]
        if x[:len(p)] != p:
            sys.exit(f'{r["id"]}: prompt tokens are not a prefix of the training tokens')
        if not span or span[-1] != turn_end_id:
            sys.exit(f'{r["id"]}: assistant span does not end with {TURN_END}')
        # the full render ends "<content><turn|>\n"; the span must match its tokens exactly
        end = content + TURN_END + '\n'
        if not full.endswith(end):
            sys.exit(f'{r["id"]}: full render does not end with the assistant content + {TURN_END}')
        f = tok(full[:-1], add_special_tokens=False)['input_ids']
        if f[-len(span):] != span:
            sys.exit(f'{r["id"]}: assistant span tokens differ from the full render\'s ending')
        if len(x) > max_len:
            dropped.append({'id': r['id'], 'reason': f'{len(x):,} tokens > {max_len:,}'})
            continue
        out.append({'id': r['id'], 'input_ids': x, 'labels': [-100] * len(p) + span})
    return out, dropped


def stats(name, exs):
    tot = sorted(len(e['input_ids']) for e in exs)
    asst = sorted(sum(l != -100 for l in e['labels']) for e in exs)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return (f'{name}: {len(exs)} examples, tokens median {q(tot, .5):,} / p99 {q(tot, .99):,} / max {tot[-1]:,}, '
            f'assistant median {q(asst, .5)} / max {asst[-1]}, {sum(tot):,} tokens total')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='local model folder (tokenizer folder is enough for --dry-run)')
    ap.add_argument('--sft', required=True, help='assemble.py output folder (train.jsonl, dev.jsonl)')
    ap.add_argument('--out', help='run folder (required unless --dry-run)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--pilot-steps', type=int, default=0)
    ap.add_argument('--load-in-4bit', action='store_true', help='QLoRA (12B, 31B); default bf16 LoRA (E4B)')
    ap.add_argument('--max-len', type=int, default=10240)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=float, default=0.03, help='warmup as a fraction of the steps')
    ap.add_argument('--grad-accum', type=int, default=16)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--eval-steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=3407)
    args = ap.parse_args()
    if not args.dry_run and not args.out:
        ap.error('--out is required unless --dry-run')
    args.model = os.path.expanduser(args.model)

    train_rows = load(os.path.join(args.sft, 'train.jsonl'))
    dev_rows = load(os.path.join(args.sft, 'dev.jsonl'))

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        model = None
    else:
        if os.path.exists(os.path.join(args.out, 'summary.json')):
            sys.exit(f'{args.out} already holds a finished run; pick another --out')
        os.makedirs(args.out, exist_ok=True)
        import unsloth  # noqa: F401  (must be imported before transformers/trl to patch them)
        from unsloth import FastModel
        model, proc = FastModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_len, dtype=None,
            load_in_4bit=args.load_in_4bit, load_in_16bit=not args.load_in_4bit,
            full_finetuning=False, trust_remote_code=False)
        tok = getattr(proc, 'tokenizer', proc)

    t0 = time.time()
    train, train_drop = build(tok, train_rows, args.max_len)
    dev, dev_drop = build(tok, dev_rows, args.max_len)
    print(f'built and checked {len(train_rows) + len(dev_rows)} examples in {time.time() - t0:.0f}s: '
          'prompt is a token prefix of every training text, every span ends with '
          f'{TURN_END} and matches the full render\'s ending')
    print(stats('train', train))
    print(stats('dev', dev))
    for d in train_drop + dev_drop:
        print(f'DROPPED {d["id"]}: {d["reason"]}')

    full_tokens = sum(len(e['input_ids']) for e in train)
    if args.pilot_steps:
        n = args.pilot_steps * args.grad_accum
        train = sorted(train, key=lambda e: -len(e['input_ids']))[:n]
        print(stats(f'pilot ({n} longest)', train))

    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    turn_end_id = tok.convert_tokens_to_ids(TURN_END)
    if args.dry_run:
        e = min(train, key=lambda e: len(e['input_ids']))
        k = next(i for i, l in enumerate(e['labels']) if l != -100)
        print(f'\nexample {e["id"]}: {len(e["input_ids"])} tokens, loss on the last {len(e["input_ids"]) - k}')
        print('--- last 40 masked tokens ⟂ trained tokens ---')
        print(tok.decode(e['input_ids'][k - 40:k]) + '⟂' + tok.decode(e['input_ids'][k:]))
        print(f'--- {TURN_END} id {turn_end_id}, bos {tok.bos_token_id}, eos {tok.eos_token_id}, '
              f'pad {tok.pad_token_id}; first token {tok.convert_ids_to_tokens(e["input_ids"][0])}')
        print(f'chat template sha {sha(tmpl)}; optimizer steps per epoch: '
              f'{len(train) // args.grad_accum} (batch 1 x accumulation {args.grad_accum})')
        return

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model = FastModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias='none',
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing='unsloth', random_state=args.seed)

    def as_ds(exs):
        return Dataset.from_list([{'input_ids': e['input_ids'], 'labels': e['labels']} for e in exs])

    pilot = bool(args.pilot_steps)
    cfg = SFTConfig(
        output_dir=os.path.join(args.out, 'checkpoints'),
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.pilot_steps if pilot else -1,
        learning_rate=args.lr, lr_scheduler_type='cosine', warmup_steps=args.warmup,
        weight_decay=0.0, optim='adamw_torch',
        max_length=args.max_len, packing=False, padding_free=False,
        eval_strategy='no' if pilot else 'steps', eval_steps=args.eval_steps,
        save_strategy='no' if pilot else 'steps', save_steps=args.eval_steps, save_total_limit=2,
        logging_steps=1, include_num_input_tokens_seen='all',
        bf16=True, seed=args.seed, data_seed=args.seed, report_to='none')

    log_path = os.path.join(args.out, 'log.jsonl')

    class Log(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            self.t0, self.tok0 = time.time(), 0
            torch.cuda.reset_peak_memory_stats()

        def on_log(self, a, state, control, logs=None, **kw):
            now = time.time()
            seen = state.num_input_tokens_seen
            rec = {'step': state.global_step, **(logs or {}),
                   'elapsed_s': round(now - self.t0, 1),
                   'tokens_per_s': round(seen / max(now - self.t0, 1e-9)),
                   'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2**30, 2)}
            with open(log_path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')

    config = {
        'args': vars(args), 'mode': 'pilot' if pilot else 'full',
        'data': {s: sha(open(os.path.join(args.sft, f'{s}.jsonl'), 'rb').read()) for s in ('train', 'dev')},
        'n_train': len(train), 'n_dev': len(dev), 'dropped': train_drop + dev_drop,
        'train_tokens': sum(len(e['input_ids']) for e in train),
        'chat_template_sha': sha(tmpl), 'turn_end_id': turn_end_id,
        'training_text': 'generation prompt (thinking off) + assistant content + <turn|>; loss on the assistant span',
        'sft_config': {k: v for k, v in cfg.to_dict().items() if isinstance(v, (int, float, str, bool, type(None)))},
        'versions': {m: __import__(m).__version__ for m in ('unsloth', 'torch', 'transformers', 'trl', 'peft')},
        'gpu': torch.cuda.get_device_name(0),
    }
    with open(os.path.join(args.out, 'config.json'), 'w') as fh:
        json.dump(config, fh, indent=1)
    print(f'chat template sha {config["chat_template_sha"]} (must match the --dry-run on the tokenizer folder)')

    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=as_ds(train),
                         eval_dataset=None if pilot else as_ds(dev), callbacks=[Log()])
    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    summary = {
        'mode': config['mode'], 'steps': trainer.state.global_step, 'train_seconds': round(secs),
        'tokens_seen': trainer.state.num_input_tokens_seen,
        'tokens_per_s': round(trainer.state.num_input_tokens_seen / secs),
        'peak_alloc_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2**30, 2),
        'gpu_total_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        'train_loss': result.training_loss,
    }
    if not pilot:
        summary['eval'] = trainer.evaluate()
        model.save_pretrained(os.path.join(args.out, 'adapter'))
        tok.save_pretrained(os.path.join(args.out, 'adapter'))
    else:
        # a full epoch at the pilot's speed (longest examples, so likely an overestimate)
        summary['est_full_epoch_hours'] = round(full_tokens / summary['tokens_per_s'] / 3600, 2)
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
