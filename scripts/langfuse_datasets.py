# ruff: noqa: E501
"""Uploads the golden and hard sets to Langfuse Cloud as datasets, and can run System 1 over them as an experiment.

    uv run --no-sync python scripts/langfuse_datasets.py                    # upload (idempotent)
    uv run --no-sync python scripts/langfuse_datasets.py --experiment b1    # + run B1 (SetFit + policy)
    uv run --no-sync python scripts/langfuse_datasets.py --experiment b1 --no-upload   # rerun only
    uv run --no-sync python scripts/langfuse_datasets.py --dry-run          # count rows, send nothing

Item ids are `<dataset>/<row id>`, so re-running updates items instead of duplicating them and removes stale ones. A new data version gets a
new `--version` (dataset name `router-<set>-<version>`). Aborts if the Presidio scan finds PII. Needs LANGFUSE_* in `.env`. The texts are public
Banking77 rows plus synthetic hard cases: no customer data.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

SETS = {"golden": ROOT / "training/golden_set/golden.jsonl", "hard": ROOT / "training/golden_set/hard_cases.jsonl"}
DESCRIPTION = {
    "golden": "Human-reviewed golden set (EN + ES). expected_output = route and urgency (0-3).",
    "hard": "Hard cases: greetings, vague, multi-intent, OOD, urgent. should_escalate says if System 1 must hand it over.",
}
METADATA_KEYS = ("intent", "category", "should_escalate", "needs_human", "needs_kb", "origin_id")


def to_item(name: str, row: dict) -> dict:
    # golden and hard share rows (the hc-* ids), and Langfuse item ids are unique per project: namespace them
    return {"id": f"{name}/{row['id']}",
            "input": {"text": row["text"], "lang": row.get("lang")},
            "expected_output": {"route": row["route"], "urgency": row["urgency"]},
            "metadata": {"set": row["split"], **{k: row[k] for k in METADATA_KEYS if k in row}}}


def prune(client, name: str, keep: set[str]) -> int:
    """Deletes items of the dataset whose id is not in `keep` (rows removed from the data, or old id scheme)."""
    stale, page = [], 1
    while True:
        batch = client.api.dataset_items.list(dataset_name=name, page=page, limit=100)
        stale += [i.id for i in batch.data if i.id not in keep]
        if page >= batch.meta.total_pages:
            break
        page += 1
    for item_id in stale:
        client.api.dataset_items.delete(item_id)
    return len(stale)


def upload(client, name: str, rows: list[dict], description: str) -> int:
    client.create_dataset(name=name, description=description)
    items = [to_item(name, row) for row in rows]
    removed = prune(client, name, {i["id"] for i in items})
    for item in items:
        client.create_dataset_item(dataset_name=name, **item)
    client.flush()
    return removed


def run_experiment(client, name: str, router_name: str, limit: int | None) -> None:
    from langfuse import Evaluation
    from run_eval import build_router

    from routercore.adapters.policy_config import load_policy, policy_path
    from routercore.adapters.route_map_config import load_route_map

    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    policy, threshold_set = load_policy(policy_path(ROOT / "config", router_name))
    router = build_router(router_name, route_map, policy)

    def task(*, item, **_):
        row = {"id": item.id, "text": item.input["text"], "lang": item.input.get("lang")}
        p = router.predict([row])[0]
        return {"route": p.route, "urgency": p.urgency, "escalated": p.escalated,
                "confidence": round(p.confidence or 0.0, 4)}

    def route_correct(*, output, expected_output, **_):
        return Evaluation(name="route_correct", value=float(output["route"] == expected_output["route"]))

    def urgency_correct(*, output, expected_output, **_):
        return Evaluation(name="urgency_correct", value=float(output["urgency"] == expected_output["urgency"]))

    def escalated(*, output, **_):
        return Evaluation(name="escalated", value=float(output["escalated"]))

    dataset = client.get_dataset(name)
    items = dataset.items[:limit] if limit else dataset.items
    result = client.run_experiment(
        name=f"{router_name}-policy", run_name=f"{router_name}-{threshold_set}",
        description="System 1 (decision model + escalation policy) over the dataset",
        data=items, task=task, evaluators=[route_correct, urgency_correct, escalated],
        max_concurrency=1, metadata={"router": router_name, "threshold_set": threshold_set})
    client.flush()
    print(result.format())


def main() -> None:
    from routercore.adapters.jsonl import read_jsonl
    from services.envfile import load_env

    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--experiment", metavar="ROUTER", help="also run this router (b1|b3|b0) as an experiment")
    parser.add_argument("--limit", type=int, help="experiment only: first N items")
    parser.add_argument("--no-upload", action="store_true", help="skip the upload, run the experiment on what is there")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_env(ROOT / ".env")

    loaded = {s: read_jsonl(SETS[s]) for s in args.sets}
    if args.dry_run:
        for s, rows in loaded.items():
            print(f"router-{s}-{args.version}: {len(rows)} items (dry run)")
        return

    from routercore.adapters.pii import PiiRedactor
    from scripts.pii_scan import scan

    redactor = PiiRedactor()
    for s, rows in loaded.items():          # PLAN §12: nothing enters a dataset without a PII scan
        if scan(rows, redactor)["strong"]:
            raise SystemExit(f"{s}: strong PII findings, upload aborted (run scripts/pii_scan.py)")

    from langfuse import get_client
    client = get_client()
    if not client.auth_check():
        raise SystemExit("Langfuse auth failed: check LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env")
    for s, rows in loaded.items():
        name = f"router-{s}-{args.version}"
        if not args.no_upload:
            removed = upload(client, name, rows, DESCRIPTION[s])
            print(f"{name}: {len(rows)} items uploaded, {removed} stale removed")
        if args.experiment:
            run_experiment(client, name, args.experiment, args.limit)


if __name__ == "__main__":
    main()
