# ruff: noqa: E501
"""Langfuse annotation queue for the judge validation, and the QA-saved measurement (PLAN §12).

A human reviews N of the 100 judge-validation cases ("was this message routed to an acceptable team?") in a Langfuse
annotation queue, blind to the Opus reference and to the judge's verdict. The score timestamps give the time per
case; the labels give the agreement of the human with the reference and with the judge.

    uv run --no-sync python scripts/langfuse_annotation.py queue --n 50            # create the queue and its traces
    uv run --no-sync python scripts/langfuse_annotation.py report --judge-seconds 1.5   # after annotating

`queue` is idempotent (same seed, same cases, same trace ids). The texts are public Banking77 rows and synthetic hard
cases; they pass the Presidio scan before upload. Needs LANGFUSE_* in `.env`.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

DIRECTORY = ROOT / "docs" / "judge_validation"
SCORE = "route_acceptable"
QUEUE = "router-qa"
CATEGORIES = {"acceptable": 1, "not_acceptable": 0}
PAUSE_CAP_S = 300        # a gap longer than this is a break, not review time


def pick_cases(cases: list[dict], n: int, seed: int) -> list[dict]:
    return sorted(random.Random(seed).sample(cases, min(n, len(cases))), key=lambda c: c["id"])


def trace_seed(case_id: str) -> str:
    return f"qa/{case_id}"


def review_seconds(times: list[float], cap: float = PAUSE_CAP_S) -> list[float]:
    """Seconds spent on each case after the first, as the gap between consecutive annotations. Gaps above `cap`
    (a break) are dropped rather than clipped, so they do not inflate the mean."""
    ordered = sorted(times)
    return [gap for a, b in zip(ordered, ordered[1:], strict=False) if (gap := b - a) <= cap]


def kappa_of(a: dict[str, int], b: dict[str, int]) -> float | None:
    from judgekit.calibration import cohen_kappa
    ids = sorted(set(a) & set(b))
    if len({a[i] for i in ids}) < 2 and len({b[i] for i in ids}) < 2:
        return None
    return round(cohen_kappa([a[i] for i in ids], [b[i] for i in ids]), 4)


def summarize(human: dict[str, int], reference: dict[str, int], judge: dict[str, int | None],
              gaps: list[float], judge_seconds: float) -> dict:
    ids = sorted(human)
    judge_ok = {i: judge[i] for i in ids if judge.get(i) is not None}
    wrong = [i for i in ids if human[i] == 0]
    flagged = [i for i in ids if judge_ok.get(i) == 0]            # the judge says "not acceptable"
    caught = [i for i in wrong if i in flagged]
    human_s = statistics.median(gaps) if gaps else None
    return {
        "n": len(ids),
        "human_not_acceptable": len(wrong),
        "kappa_human_vs_opus": kappa_of(human, reference),
        "kappa_human_vs_judge": kappa_of(human, judge_ok),
        "kappa_judge_vs_opus_same_cases": kappa_of(judge_ok, {i: reference[i] for i in ids}),
        "human_seconds_per_case_median": round(human_s, 1) if human_s is not None else None,
        "human_seconds_per_case_mean": round(statistics.fmean(gaps), 1) if gaps else None,
        "gaps_used": len(gaps),
        "judge_seconds_per_case": judge_seconds,
        "qa_time_saved_pct": round(100 * (1 - judge_seconds / human_s), 1) if human_s else None,
        "review_only_judge_rejected": {
            "share_of_cases_reviewed": round(len(flagged) / len(ids), 3) if ids else None,
            "real_errors_caught": f"{len(caught)}/{len(wrong)}",
            "time_saved_pct": round(100 * (1 - len(flagged) / len(ids)), 1) if ids else None,
        },
    }


def connect():
    from langfuse import get_client

    from services.envfile import load_env
    load_env(ROOT / ".env")
    client = get_client()
    if not client.auth_check():
        raise SystemExit("Langfuse auth failed: check LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env")
    return client


def ensure_config(client) -> str:
    from langfuse.api.commons.types import ConfigCategory
    for config in client.api.score_configs.get(limit=100).data:
        if config.name == SCORE and not config.is_archived:
            return config.id
    created = client.api.score_configs.create(
        name=SCORE, data_type="CATEGORICAL",
        categories=[ConfigCategory(label=label, value=value) for label, value in CATEGORIES.items()],
        description="Is the message routed to an acceptable team (the right one, or one that would clearly handle it)?")
    return created.id


def ensure_queue(client, config_id: str, description: str) -> str:
    for queue in client.api.annotation_queues.list_queues(limit=100).data:
        if queue.name == QUEUE:
            return queue.id
    return client.api.annotation_queues.create_queue(
        name=QUEUE, score_config_ids=[config_id], description=description).id


def queued_trace_ids(client, queue_id: str) -> set[str]:
    found, page = set(), 1
    while True:
        batch = client.api.annotation_queues.list_queue_items(queue_id, page=page, limit=100)
        found |= {item.object_id for item in batch.data}
        if page >= batch.meta.total_pages:
            return found
        page += 1


def cmd_queue(args) -> None:
    from routercore.adapters.jsonl import read_jsonl
    from routercore.adapters.pii import PiiRedactor
    from routercore.adapters.route_map_config import load_route_map
    from scripts.pii_scan import scan

    cases = pick_cases(read_jsonl(DIRECTORY / "cases.jsonl"), args.n, args.seed)
    if scan([{"id": c["id"], "text": c["text"]} for c in cases], PiiRedactor())["strong"]:
        raise SystemExit("strong PII findings, upload aborted (run scripts/pii_scan.py)")
    routes = load_route_map(ROOT / "config" / "route_map.yaml").routes
    description = ("For each message, mark whether the team it was routed to is acceptable: the team primarily "
                   "responsible for the customer's main request, or one that would clearly handle it. Teams: "
                   + "; ".join(f"{name} = {text}" for name, text in routes.items()))

    client = connect()
    config_id = ensure_config(client)
    queue_id = ensure_queue(client, config_id, description)
    for case in cases:
        trace_id = client.create_trace_id(seed=trace_seed(case["id"]))
        with client.start_as_current_observation(
                trace_context={"trace_id": trace_id}, name="qa-case", input={"message": case["text"]},
                output={"routed_to": case["proposed_route"]}, metadata={"case_id": case["id"]}):
            pass
    client.flush()

    have = queued_trace_ids(client, queue_id)
    added = 0
    for case in cases:
        trace_id = client.create_trace_id(seed=trace_seed(case["id"]))
        if trace_id in have:
            continue
        for attempt in range(6):                 # the trace is ingested asynchronously
            try:
                client.api.annotation_queues.create_queue_item(queue_id, object_id=trace_id, object_type="TRACE")
                added += 1
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5)
    print(f"queue '{QUEUE}' ({queue_id}): {len(cases)} cases, {added} added. "
          f"Open Langfuse > Human Annotation > {QUEUE} and score '{SCORE}' on each.")


def fetch_annotations() -> list[dict]:
    """Annotation scores through the v3 scores endpoint (the SDK still calls v2, which Langfuse Cloud removed
    on 2026-11-16). Each row has `subject.id` (the trace), `value` (the category) and `createdAt`."""
    import os

    import requests
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    scores, cursor = [], None
    while True:
        params = [("name", SCORE), ("source", "ANNOTATION"), ("limit", 100), ("fields", "core"), ("fields", "subject")]
        if cursor:
            params.append(("cursor", cursor))
        reply = requests.get(os.environ["LANGFUSE_HOST"].rstrip("/") + "/api/public/v3/scores", params=params,
                             auth=auth, timeout=30)
        reply.raise_for_status()
        body = reply.json()
        scores += body["data"]
        cursor = body.get("meta", {}).get("cursor")
        if not cursor:
            return scores


def cmd_report(args) -> None:
    from datetime import datetime

    from routercore.adapters.jsonl import read_jsonl

    client = connect()
    expected = {client.create_trace_id(seed=trace_seed(c["id"])): c["id"]
                for c in read_jsonl(DIRECTORY / "cases.jsonl")}
    latest: dict[str, dict] = {}
    for score in sorted(fetch_annotations(), key=lambda s: s["createdAt"]):
        trace_id = (score.get("subject") or {}).get("id")
        if trace_id in expected:
            latest[expected[trace_id]] = score          # a corrected annotation replaces the first
    if not latest:
        raise SystemExit("no annotations yet")
    human = {case_id: CATEGORIES[s["value"]] for case_id, s in latest.items()}
    gaps = review_seconds([datetime.fromisoformat(s["createdAt"].replace("Z", "+00:00")).timestamp()
                           for s in latest.values()])

    reference = {r["id"]: int(r["acceptable"]) for r in read_jsonl(DIRECTORY / "reference.jsonl")}
    judge = json.loads((DIRECTORY / args.judge_file).read_text(encoding="utf-8"))
    report = summarize(human, reference, judge, gaps, args.judge_seconds)
    if args.model_labels:                  # cases the human did not annotate, labelled blind by a model
        model = {r["id"]: int(r["acceptable"]) for r in read_jsonl(Path(args.model_labels))}
        both = {**{i: v for i, v in model.items() if i not in human}, **human}
        mixed = summarize(both, reference, judge, [], args.judge_seconds)
        report["human_plus_model_labels"] = {
            "n": mixed["n"], "human": len(human), "model": mixed["n"] - len(human),
            "model_labels": args.model_labels_name,
            "kappa_labels_vs_opus_reference": mixed["kappa_human_vs_opus"],
            "kappa_labels_vs_judge": mixed["kappa_human_vs_judge"],
            "review_only_judge_rejected": mixed["review_only_judge_rejected"]}
    report["judge"] = args.judge_file
    out = ROOT / "docs" / "results" / "qa_saved.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [json.dumps({"id": i, "acceptable": v}) for i, v in sorted(human.items())]
    (DIRECTORY / "human.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("queue")
    queue.add_argument("--n", type=int, default=50)
    queue.add_argument("--seed", type=int, default=0)
    report = commands.add_parser("report")
    report.add_argument("--judge-file", default="judge_gpt-oss-120b-r2.json")
    report.add_argument("--judge-seconds", type=float, required=True, help="judge time per case, from the gateway log")
    report.add_argument("--model-labels", help="JSONL {id, acceptable} for the cases the human did not annotate")
    report.add_argument("--model-labels-name", default="opus-5.5-medium-blind")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cmd_queue(args) if args.command == "queue" else cmd_report(args)


if __name__ == "__main__":
    main()
