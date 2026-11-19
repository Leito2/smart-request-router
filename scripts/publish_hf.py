"""Stage and publish the dataset and the SetFit System 1 to the Hugging Face Hub (PLAN M9).

Usage: python scripts/publish_hf.py --user WhiteUniversal            # stage only, prints what would go up
       python scripts/publish_hf.py --user WhiteUniversal --upload --only model --message "..."   # one repo
       python scripts/publish_hf.py --user WhiteUniversal --upload   # create PUBLIC repos and upload
Upload needs `hf auth login` first.
Repos: <user>/smart-request-router-dataset, <user>/smart-request-router-setfit and the Space <user>/smart-request-router.
"""
# ruff: noqa: E501  (the Hub cards below are prose that the Hub wraps itself)
import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_FRONT = """---
license: cc-by-4.0
language: [en, es]
task_categories: [text-classification]
pretty_name: Smart Request Router - banking support routing
size_categories: [10K<n<100K]
source_datasets: [PolyAI/banking77]
tags: [banking, intent-detection, routing, urgency, multilingual]
---

"""
DATASET_NOTE = """

## Files
`dataset.jsonl` (train/val/test), `golden.jsonl` (497 reviewed rows), `golden_reviews.jsonl` (reviewer verdicts),
`hard_cases.jsonl` (394), `MANIFEST.json` (row counts and hashes). Spanish rows were machine translated with Gemma 3 4B
(subject to the Gemma terms of use). Built from PolyAI/banking77 (CC BY 4.0): attribution to Casanueva et al., 2020 is required.
Code: https://github.com/Leito2/smart-request-router
"""
MODEL_CARD = """---
license: apache-2.0
language: [en, es]
library_name: setfit
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
datasets: [{user}/smart-request-router-dataset]
tags: [setfit, text-classification, banking, routing]
---

# Smart Request Router - SetFit System 1 (B1)

A SetFit route head (`intent/` -> 12 routes) and an urgency head (`urgency/` -> 0-3: a logistic regression over frozen
`paraphrase-multilingual-MiniLM-L12-v2` embeddings that predicts the Banking77 intent and maps it to its urgency rule; a
multi-sentence message is also scored sentence by sentence and the most urgent part wins), trained on the dataset above. It is the **System 1** of https://github.com/Leito2/smart-request-router: it answers when confident and
escalates otherwise (the escalation policy lives in the repo, `config/policy.setfit.yaml`).

Test split (3,311 rows, no escalation): accuracy 0.767, macro-F1 0.694, urgent recall 0.923. With the escalation policy
(`policy.yaml`, 600-row test sample): selective accuracy 0.874 at 66% coverage, urgent recall 1.0; golden set (497): 0.883 at
60%, urgent recall 1.0. System 1 p95 is about 50 ms on CPU.

## Use
```python
from huggingface_hub import snapshot_download
path = snapshot_download("{user}/smart-request-router-setfit")
# then, from the repo: SetFitRouter.load(path)  (routercore.adapters.setfit_model)
# the urgency head needs scikit-learn, joblib and sentence-transformers (the `ml` group)
```
Limits: English and Spanish only; 12 banking-support routes; Spanish training data is machine translated; urgency labels
are rule-based from the intent, not human-annotated.
"""


def stage(user: str, out: Path) -> tuple[Path, Path, Path]:
    dataset, model, space = out / "dataset", out / "setfit", out / "space"
    dataset.mkdir(parents=True), model.mkdir(parents=True)
    for source, name in [("training/data/dataset.jsonl", "dataset.jsonl"),
                         ("training/golden_set/golden.jsonl", "golden.jsonl"),
                         ("training/golden_set/hard_cases.jsonl", "hard_cases.jsonl"),
                         ("training/golden_set/reviews.jsonl", "golden_reviews.jsonl"),
                         ("training/golden_set/MANIFEST.json", "MANIFEST.json")]:
        shutil.copy(ROOT / source, dataset / name)
    card = (ROOT / "docs" / "dataset_card.md").read_text(encoding="utf-8")
    (dataset / "README.md").write_text(DATASET_FRONT + card + DATASET_NOTE, encoding="utf-8")
    shutil.copytree(ROOT / "models" / "b1_setfit", model, dirs_exist_ok=True)
    (model / "README.md").write_text(MODEL_CARD.replace("{user}", user), encoding="utf-8")
    shutil.copy(ROOT / "config" / "policy.setfit.yaml", model / "policy.yaml")   # the Space reads it from the model repo
    shutil.copytree(ROOT / "space", space)
    app = (space / "app.py").read_text(encoding="utf-8").replace("WhiteUniversal", user)
    (space / "app.py").write_text(app, encoding="utf-8")
    (space / "README.md").write_text((space / "README.md").read_text(encoding="utf-8").replace("WhiteUniversal", user),
                                     encoding="utf-8")
    return dataset, model, space


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--upload", action="store_true", help="create public repos and upload")
    parser.add_argument("--only", choices=["dataset", "model", "space"], help="upload just this repo")
    parser.add_argument("--message", default="Initial upload", help="commit message on the Hub")
    parser.add_argument("--space", action="store_true",
                        help="also publish the Gradio Space (the Hub needs a PRO plan for it: HTTP 402 on free accounts)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        dataset, model, space = stage(args.user, Path(tmp))
        for folder in (dataset, model, space):
            size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
            print(f"staged {folder.name}: {size:.1f} MB")
        if not args.upload:
            return
        from huggingface_hub import HfApi
        api = HfApi()
        for name, kind, folder in [("smart-request-router-dataset", "dataset", dataset),
                                   ("smart-request-router-setfit", "model", model),
                                   *([("smart-request-router", "space", space)] if args.space else [])]:
            if args.only and kind != args.only:
                continue
            repo = f"{args.user}/{name}"
            extra = {"space_sdk": "gradio"} if kind == "space" else {}
            api.create_repo(repo, repo_type=kind, private=False, exist_ok=True, **extra)
            # a model republished with another urgency head must not keep the old head's files on the Hub
            stale = {"delete_patterns": ["urgency/*"]} if kind == "model" else {}
            api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=kind,
                              commit_message=args.message, **stale)
            prefix = {"dataset": "datasets/", "space": "spaces/", "model": ""}[kind]
            print(f"uploaded https://huggingface.co/{prefix}{repo}")


if __name__ == "__main__":
    main()
