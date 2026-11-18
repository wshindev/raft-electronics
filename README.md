# RAFT-001: does retrieval-augmented fine-tuning help a small model on technical material?

This project builds RAFT (Retrieval-Augmented Fine-Tuning) training data and an eval set from an open electronics curriculum, fine-tunes a small open model on one 32 GB GPU, and measures the result.

**Short answer: not on this material.** Three LoRA adapters for Gemma 4 E4B, each trained on answers from a different teacher model, all scored below the untuned base model. Retrieval on its own gave the gain. The likely reason, which we did not test, is that the base model already knows this public coursework, so fine-tuning had little to add and still cost some general skill. The next experiment is a niche source the model hasn't seen.

Source: [*Modular Electronics Learning*](https://ibiblio.org/kuphaldt/socratic/model/index.html) (ModEL) by Tony R. Kuphaldt, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Write-up: [wshin.dev/lab/raft-electronics](https://wshin.dev/lab/raft-electronics/)

Eval set: [huggingface.co/datasets/wshindev/model-electronics-eval](https://huggingface.co/datasets/wshindev/model-electronics-eval)

## What RAFT is
Each training example gives the model a question and 5 documents. Usually one document holds the answer and the other four are decoys; sometimes none does. The model learns to find the right document, quote it, and reason to an answer. The idea is that a model trained this way uses retrieved documents better than one that only receives them at answer time.

## Pipeline
One script per step in `pipeline/`.

| step | script | what it does |
|---|---|---|
| 01 | `parse.py` | LaTeX → 5,846 deduplicated sections, 1,249 exam Q/A pairs, variant groups from the exam generator source |
| 02 | `render.py` | 7,619 EPS diagrams → PNG, with text labels pulled from the diagram files |
| 03 | `describe.py` | figure descriptions with a vision model (piloted, then paused; the first run is text-only) |
| 04 | `clean_text.py` | LaTeX → readable text, figures as label placeholders |
| 05 | `split.py` | held-out modules; 243 exam items that need no figure; variant groups never split |
| 06 | `generate.py`, `embed.py`, `assemble.py` | RAFT Q/A generation, embeddings, hard decoys (dense + BM25), decontamination against the exam |
| 07 | `eval_sets.py`, `train.py`, `eval_run.py`, `score.py`, `keypoints.py`, `judge.py`, `teacher.py` | eval sets, LoRA training, inference via llama-server, key-point grading, local teacher |

See `docs/pipeline_overview.md` for a flowchart and file-level detail.

## Eval
- **Exam:** 243 of Kuphaldt's own exam questions (31 groups of parametric variants). These need problem-solving; no document in the corpus holds the answer.
- **Held-out questions:** 736 questions written by a separate model (Fable 5.1) from 16 modules kept out of training. Each has a known source document.
- Each set runs **without documents** and **with the top 5** from the eval retriever (Qwen3-Embedding-8B + BM25 → Qwen3-Reranker-8B).
- **Grading:** key points are written once per item. A judge model marks each point covered, partly covered or missing in the final answer, and Python checks numbers with a tolerance. Two judges were validated against 60 hand-graded answers: Claude Haiku 5.5 (κ 0.81) and a local Gemma 4 31B (κ 0.78). A separate check tests whether the model's quotes appear word for word in the documents.

## Results (Gemma 4 E4B-it, LoRA r16, 1 epoch)
Adapter minus base, paired on the same items, with 95% bootstrap intervals in brackets:

| training answers by | judge | exam, no docs | exam, top 5 | held-out, no docs | held-out, top 5 |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | Haiku | −0.08 [−0.16, +0.03] | −0.20 [−0.26, −0.13] | −0.09 [−0.11, −0.07] | −0.05 [−0.07, −0.04] |
| Claude Opus 5.5 | Haiku | −0.10 [−0.16, −0.04] | −0.19 [−0.25, −0.13] | −0.08 [−0.10, −0.06] | −0.06 [−0.07, −0.04] |
| Gemma 4 31B (local) | 31B | −0.05 [−0.10, +0.00] | −0.07 [−0.14, +0.02] | −0.03 [−0.05, −0.01] | −0.04 [−0.06, −0.03] |

Base scores (31B judge): exam 0.80 / 0.82, held-out 0.74 / 0.91. Scores from different judges are not comparable with each other.

What was measured along the way:
- **Short answers:** the Claude-written training answers were brief, and the adapters copied that length. Telling the base model to be brief reproduces most of the gap on held-out questions with documents (Test A).
- **Invented quotes:** the first adapters invented quotes on the exam (54–58% of their quotes were found in no document). With 31B-written training data this fell to 11%.
- **Not a serving bug:** the served GGUF adapter behaves like the trained one (`serve_check.py`).
- **Not missing data:** the training data covers the topics the adapters failed on, with correct examples.

What was not varied: model size (E4B only), training strength, the training mix, and the source.

## What carries over to a new source
- An all-local pipeline: teacher, retriever, reranker and judge all run on llama-server.
- A judge validated against hand grades.
- **A headroom check before generating anything:** run the base model with and without documents on 20–30 questions. Fine-tune only if the base model is weak without documents.

## Running it
- Python 3.14 via [uv](https://docs.astral.sh/uv/) for the pipeline; a separate `train/` uv project (Python 3.12, Unsloth) for training.
- Ghostscript (`gs -dSAFER`) for rendering; [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` for embeddings, reranking, inference and the local judge.
- Steps that call the Claude API read `ANTHROPIC_API_KEY` from the environment.
- `data/` is not included; it regenerates from the ModEL source plus `pipeline/`. Get ModEL from the author's site and pass its paths as arguments.

Scripts run in Python isolated mode (`-I`) because they read untrusted source files. Paths are passed as arguments; `HOST` is your llama-server, and `<models>` is where you keep downloaded Hugging Face model folders. `LLAMA_API_KEY` is read if the server needs one.

**01–05: parse, render, clean, split** (no models)
```
uv run python3 -I pipeline/parse.py --content ModEL/content --exammake ModEL/exammake/exammake.c --source model --out data
uv run python3 -I pipeline/render.py --refs data/model/image_refs.jsonl --eps-dir ModEL/eps
uv run python3 -I pipeline/clean_text.py --data data/model
uv run python3 -I pipeline/split.py --data data/model
```
Figure descriptions (paused, Claude API):
```
uv run python3 -I pipeline/pilot_select.py --refs data/model/image_refs.jsonl --meta data/model/image_meta.jsonl --pairs data/model/exam_pairs.jsonl
uv run python3 -I pipeline/describe.py {estimate|run|review} --refs data/model/image_refs.jsonl --images data/model/pilot/pilot_images.jsonl [--out data/model/pilot]
```

**06: RAFT generation, embeddings, assembly**
```
# Q/A generation (Claude API; estimate and review are offline). Pilot first, then Batch API for the full split.
uv run python3 -I pipeline/generate.py {estimate|run|review} --data data/model [--limit 20] [--out data/model/raft_pilot]
uv run python3 -I pipeline/generate.py {submit|status|collect} --data data/model [--split train] --out data/model/raft_train

# embeddings via llama-server
uv run python3 -I pipeline/embed.py {check|run} --data data/model --url http://HOST:8080 --gen raft_train raft_heldout_fable [--model-tag qwen3-emb-8b-f16]

# RAFT examples with hybrid (dense + BM25) decoys, decontaminated against the exam
uv run python3 -I pipeline/assemble.py --data data/model --gen raft_train --emb data/model/embeddings/qwen3-emb-8b-f16 --out data/model/sft_sonnet_hybrid --bm25
```

**07a: retriever choice and eval sets**
```
uv run python3 -I pipeline/retrieval_eval.py --data data/model --gen raft_heldout_fable --emb data/model/embeddings/<tag> ... [--bm25] [--dump-candidates 20] [--rerank data/model/rerank/<tag>/raft_heldout_fable.jsonl ...]
uv run python3 -I pipeline/rerank.py {check|run} --data data/model --url http://HOST:8080 --cands data/model/rerank/candidates_raft_heldout_fable.jsonl [--model-tag qwen3-rerank-4b-bf16]   # llama-server --reranking
uv run python3 -I pipeline/eval_sets.py candidates --data data/model --gen raft_heldout_fable --emb data/model/embeddings/qwen3-emb-8b-f16
uv run python3 -I pipeline/eval_sets.py build --data data/model --gen raft_heldout_fable --rerank data/model/rerank/elec-rerank-8B-q8_0 --out data/model/eval_sets
```

**07b: training** (`train/` venv, GPU; `--dry-run` runs on CPU with only the tokenizer)
```
train/.venv/bin/python -I pipeline/template_check.py --tokenizer train/tokenizers/gemma-4-E4B-it --sft data/model/sft_sonnet_hybrid --out data/model/template_check
train/.venv/bin/python -I pipeline/train.py --model <models>/gemma-4-E4B-it --sft data/model/sft_sonnet_hybrid --out data/model/runs/e4b_sonnet [--dry-run] [--pilot-steps 20] [--load-in-4bit]
```
Adapters were converted to GGUF by hand with llama.cpp's converter (not part of `pipeline/`) and served on the Q8_0 base with `--lora`.

**07c: local teacher** (a served model rewrites the answers of an existing RAFT set, checked against the reference answers)
```
uv run python3 -I pipeline/teacher.py {run|check|assemble|report} --sft data/model/sft_sonnet_hybrid --out data/model/teacher/gemma31b [--url http://HOST:8080 --split train --sample 100] [--to data/model/sft_gemma31b]
```

**07d: eval runs** (llama-server with `--jinja`; use `-np 1` for reproducible greedy runs)
```
uv run python3 -I pipeline/eval_run.py {check|run} --url http://HOST:8080 --config e4b_base --inputs data/model/eval_sets/*.jsonl --out data/model/eval_runs [--limit 20] [--lora-scale 0.5] [--workers 1]
train/.venv/bin/python -I pipeline/serve_check.py --model <models>/gemma-4-E4B-it --adapter data/model/runs/e4b_sonnet/adapter --input data/model/eval_sets/exam_nodocs.jsonl --served data/model/eval_runs/e4b_sonnet/exam_nodocs.jsonl --served-base data/model/eval_runs/e4b_base/exam_nodocs.jsonl --ids ... --out data/model/serve_check/e4b_sonnet
```

**07e: scoring**
```
uv run python3 -I pipeline/score.py extract --runs data/model/eval_runs --out data/model/scores [--sets data/model/eval_sets]
uv run python3 -I pipeline/score.py ground --runs data/model/eval_runs --sets data/model/eval_sets --out data/model/scores

# key points, once per eval item (Claude API; estimate, review and check are offline)
uv run python3 -I pipeline/keypoints.py {estimate|run|submit|status|collect|review|check} --data data/model --gen raft_heldout_fable [--limit 20 --ids ...] --out data/model/keypoints

# judge: Claude (API) or a local model on llama-server
uv run python3 -I pipeline/judge.py {estimate|run|submit|status|collect} --data data/model --gen raft_heldout_fable --scores data/model/scores --keypoints data/model/keypoints --judge {haiku|sonnet} --out data/model/scores/judge/haiku
uv run python3 -I pipeline/judge.py run ... --configs e4b_base e4b_gemma31b --judge local --url http://HOST:8080 --workers 1 --out data/model/scores/judge/gemma31b

# judge validation: hand grades and agreement (offline; --sample 204 needs --configs e4b_base e4b_opus e4b_sonnet)
uv run python3 -I pipeline/judge.py {grade|agree} ... --sample 204 [--n 20 --out .../human.jsonl | --a DIR --b DIR --human F]

# score table and paired comparisons (offline)
uv run python3 -I pipeline/judge.py report ... --out data/model/scores/judge/gemma31b --grounding data/model/scores/grounding
```
In the `judge.py` lines, `...` stands for the shared `--data --gen --scores --keypoints` arguments of the first one. Before any full Claude API run, pilot on a small sample and check the cost estimate.

**08: eval set export** (offline; one row per question, plus corpus and prompts; card in `docs/eval_set_card.md`)
```
uv run python3 -I pipeline/export_eval.py --data data/model --gen raft_heldout_fable --sets data/model/eval_sets --keypoints data/model/keypoints --out data/model/eval_release
```

## License and attribution
- Source content: *Modular Electronics Learning* by Tony R. Kuphaldt, CC BY 4.0 (`license/`). Any data derived from it carries that attribution.
- Gemma 4: Apache 2.0 (`license/Gemma-Apache-4-License.md`).
- Code: Apache 2.0 (`LICENSE`).
