# Dataset card — Smart Request Router (M1)

| | |
|---|---|
| **Source (English)** | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), 13,083 bank customer queries, 77 intents. **License: CC BY 4.0** (Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence Encoders*). Attribution required; derived work may be shared under the same terms. |
| **Spanish** | A stratified sample (20 per intent from train, 3 per intent from test) translated by a local `gemma3:4b` through Ollama with a strict few-shot prompt (`training/translate_es.py`). Machine-translated: expect small errors. |
| **Hard cases** | Written for this project (`training/hard_cases.py`): greetings, very short messages, out-of-domain, negations, sarcasm; plus multi-intent messages composed from two Banking77 test messages of different routes. Evaluation only: never used for training. |
| **Labels** | 77 intents → 12 routes (`config/route_map.yaml`). `urgency` 0–3, `needs_human` and `needs_kb` are rule-based from the intent (`training/data_prep.py`), not hand-labelled. |

## Splits and leakage rules
- Banking77's official test split is our `test`. The official train split is divided into `train` / `val` (10%) by hash of the **origin id**, so a message and its Spanish translation always share a split.
- Any train/val text that also appears in test (case, punctuation and spacing ignored) is dropped.
- Verified on the built data: 0 origins in more than one split, 0 texts shared between test and train/val.

## Golden set (500)
340 regular test rows balanced by (route, language) + 160 hard cases balanced by category. Candidates are in `training/golden_set/golden.jsonl`; each row has a `review` field that is **still empty**: the frontier-model review has not been run (see `training/golden_set/REVIEW.md` and `training/apply_review.py`). The kappa reference set is separate and arrives with `judgekit` (M7b).

## Known limits
- Banking77 has no `general` or out-of-domain examples, so `training/general_train.py` adds ~100 greetings, thanks, tiny and off-topic messages (train/val only). A test keeps them disjoint from the evaluation hard cases.
- Urgent rows in the golden set are few (~35); urgent recall is measured on the whole test split.
- Spanish rows are machine translated; only a reviewed sample is trusted as ground truth.
