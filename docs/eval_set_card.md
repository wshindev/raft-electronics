---
license: cc-by-4.0
language:
- en
pretty_name: ModEL electronics eval set
size_categories:
- n<1K
task_categories:
- question-answering
tags:
- electronics
- rag
- retrieval
- evaluation
configs:
- config_name: exam
  data_files: exam.jsonl
- config_name: fable
  data_files: fable.jsonl
- config_name: corpus
  data_files: corpus.jsonl
---

# ModEL electronics eval set

Questions on electronics coursework, for testing small models with and without retrieved documents. Built from Tony R. Kuphaldt's *Modular Electronics Learning* (ModEL, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) as part of RAFT-001 ([code](https://github.com/wshindev/raft-electronics), [write-up](https://wshin.dev/lab/raft-electronics/)).

## Contents

| file | rows | what |
|---|---|---|
| `exam.jsonl` | 243 | Kuphaldt's own exam questions and answers that need no figure; 42 parametric variant groups, scored as 31 groups |
| `fable.jsonl` | 736 | questions written by Claude Fable 5.1 from 16 ModEL modules (319 source sections), each with a reference answer and its source section |
| `corpus.jsonl` | 5,846 | every ModEL section used for retrieval (clean text; figures as placeholders with their labels) |
| `prompts.json` | — | the exact prompts used, with and without documents |

Each question row holds:
- `question`, `reference`: exam answer (`answer`), or for Fable a `reasoning` with verbatim quotes and a `final_answer`.
- `key_points`: what a correct final answer must contain, as a list. `kind` is `numeric` (values with units and a tolerance), `exact` (accepted forms) or `conceptual` (a statement).
- `notes` (some rows): `ungradable` (part of the question can't be checked in text, e.g. "sketch the circuit") or `reference_issue` (an error in the reference; the key points accept the correct form, and usually the reference's as well).
- `retrieved`: the eval retriever's top 5 section ids, best first. The retriever is Qwen3-Embedding-8B + BM25, fused, top 20 → Qwen3-Reranker-8B.
- `score_group`: report scores as a mean over groups (exam: merged variant groups) or over source sections (Fable), and bootstrap over these.
- Exam: `variant_group`, `topics`. Fable: `answer_type`, `source_chunk`, `module`, `source_chunk_retrieved`.
- `provenance`: who wrote the question, the reference and the key points.

## How it was scored in RAFT-001
Each condition (no documents / top 5) is one prompt per question, built from `prompts.json`. The model's final answer is the text after the last `<ANSWER>` marker. A judge model marks each key point covered / partly / missing, and numeric values are compared in code against the tolerance. The scorer is [`pipeline/judge.py`](https://github.com/wshindev/raft-electronics/blob/main/pipeline/judge.py) in the repo. Judges were validated against 60 hand-graded answers: Claude Haiku 5.5 (κ 0.81), Gemma 4 31B (κ 0.78, slightly lenient on conceptual partial credit).

Baseline, Gemma 4 E4B-it (31B judge): exam 0.80 without documents / 0.82 with top 5; Fable 0.74 / 0.91.

## Provenance and licence
- Exam questions and answers, corpus text: Tony R. Kuphaldt, *Modular Electronics Learning*, CC BY 4.0. Cleaned from LaTeX by this project (`pipeline/parse.py`, `clean_text.py`).
- Fable questions and reference answers: Claude Fable 5.1. Key points: Claude Opus 5.5. Both checked by script against the source text and the reference answers. Not all were hand-reviewed.
- Retrieval rankings: Qwen3-Embedding-8B and Qwen3-Reranker-8B, run locally.
- Attribution: when using this set, credit Tony R. Kuphaldt's *Modular Electronics Learning* (CC BY 4.0) and note the changes above.

## Known limitations
- **Likely seen in pretraining.** ModEL is public, so a model may already know this material. Base E4B scores high, which leaves little headroom.
- **The exam is text-only.** 1,006 of the 1,249 exam questions depend on a figure and are left out. Some kept questions still ask for a sketch (marked `ungradable`).
- **Exam answers are not in the corpus.** These questions test working things out, not lookup. Retrieved documents help little (+0.02 for base E4B).
- **Fable source sections are in the corpus.** They were held out from RAFT-001's training data only. The top 5 contains the source section for 98.4% of the questions.
- **LaTeX remains** in exam questions and answers (e.g. `k$\Omega$`).
- **Model-written key points.** These were checked by script and partly by hand. Conceptual points are where judges disagree most.
