"""Batch evaluation job over versioned datasets in MinIO/S3 (PLAN §12 industrial evaluation).

The golden set lives in a versioned bucket; each run reads the dataset, evaluates one router and writes a
report that cites the exact object version it used, so a result traces back to the data that produced it.

Usage: python eval/batch_job.py --router b0 [--dataset golden/golden.jsonl]
Env: S3_ENDPOINT (http://localhost:9000), S3_KEY, S3_SECRET, S3_BUCKET (router-eval)
"""
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from harness import evaluate

ROOT = Path(__file__).resolve().parent.parent


def load_dataset(s3, bucket: str, key: str) -> tuple[list[dict], str | None]:
    response = s3.get_object(Bucket=bucket, Key=key)
    lines = response["Body"].read().decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line], response.get("VersionId")


def ensure_bucket(s3, bucket: str) -> None:
    """Versioned bucket: every upload of the dataset keeps its own VersionId, which the reports cite."""
    try:
        s3.create_bucket(Bucket=bucket)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})


def upload_dataset(s3, bucket: str, key: str, path: Path) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())


def report_key(router: str, dataset: str, timestamp: str) -> str:
    return f"reports/{timestamp}-{router}-{dataset}.json"


def run_batch(s3, router, *, router_name: str, dataset_key: str, bucket: str, now: str | None = None) -> str:
    rows, version = load_dataset(s3, bucket, dataset_key)
    metrics = evaluate(rows, router.predict(rows))
    timestamp = now or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    report = {"router": router_name, "dataset": dataset_key, "dataset_version": version, "rows": len(rows),
              "metrics": {k: round(v, 4) for k, v in metrics.items()}}
    key = report_key(router_name, Path(dataset_key).stem, timestamp)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(report, indent=2).encode("utf-8"))
    return key


def main() -> None:
    import boto3

    sys.path.insert(0, str(ROOT / "training"))
    from run_eval import build_router

    from routercore.adapters.policy_config import load_policy
    from routercore.adapters.route_map_config import load_route_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="b0")
    parser.add_argument("--dataset", default="golden/golden.jsonl")
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--seed", action="store_true", help="create the versioned bucket, upload golden set")
    args = parser.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
                      aws_access_key_id=os.environ.get("S3_KEY", "minioadmin"),
                      aws_secret_access_key=os.environ.get("S3_SECRET", "minioadmin"))
    bucket = os.environ.get("S3_BUCKET", "router-eval")
    if args.seed:
        ensure_bucket(s3, bucket)
        upload_dataset(s3, bucket, args.dataset, ROOT / "training" / "golden_set" / "golden.jsonl")
    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    policy, _ = load_policy(ROOT / "config" / "policy.yaml")
    router = build_router(args.router, route_map, policy if args.policy else None)
    key = run_batch(s3, router, router_name=args.router, dataset_key=args.dataset,
                    bucket=os.environ.get("S3_BUCKET", "router-eval"))
    print(f"report written to s3://{os.environ.get('S3_BUCKET', 'router-eval')}/{key}")


if __name__ == "__main__":
    main()
