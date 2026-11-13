"""Stage and publish the dataset and the SetFit System 1 to the Hugging Face Hub (PLAN M9).

Usage: python scripts/publish_hf.py --user WhiteUniversal            # stage only, prints what would go up
       python scripts/publish_hf.py --user WhiteUniversal --upload   # create PUBLIC repos and upload
Upload needs `hf auth login` first.
Repos: <user>/smart-request-router-dataset and <user>/smart-request-router-setfit.
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

Two SetFit models (`intent/` -> 12 routes, `urgency/` -> 0-3) on `paraphrase-multilingual-MiniLM-L12-v2`, trained on the
dataset above. It is the **System 1** of https://github.com/Leito2/smart-request-router: it answers when confident and
escalates otherwise (the escalation policy lives in the repo, `config/policy.setfit.yaml`).

Test split (3,080 rows, no escalation): accuracy 0.778, macro-F1 0.704, urgent recall 0.879. With the escalation policy
(600-row test sample): selective accuracy 0.920 at 50% coverage, urgent recall 0.956.

## Use
```python
from huggingface_hub import snapshot_download
path = snapshot_download("{user}/smart-request-router-setfit")
# then, from the repo: SetFitRouter.load(path)  (training/baselines/setfit_model.py)
```
Limits: English and Spanish only; 12 banking-support routes; Spanish training data is machine translated; urgency labels
are rule-based from the intent, not human-annotated.
"""


def stage(user: str, out: Path) -> tuple[Path, Path]:
    dataset, model = out / "dataset", out / "setfit"
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
    return dataset, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--upload", action="store_true", help="create public repos and upload")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        dataset, model = stage(args.user, Path(tmp))
        for folder in (dataset, model):
            size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
            print(f"staged {folder.name}: {size:.1f} MB")
        if not args.upload:
            return
        from huggingface_hub import HfApi
        api = HfApi()
        for name, kind, folder in [("smart-request-router-dataset", "dataset", dataset),
                                   ("smart-request-router-setfit", "model", model)]:
            repo = f"{args.user}/{name}"
            api.create_repo(repo, repo_type=kind, private=False, exist_ok=True)
            api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=kind,
                              commit_message="Initial upload")
            print(f"uploaded https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
