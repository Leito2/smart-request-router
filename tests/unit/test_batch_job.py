import json

from batch_job import load_dataset, report_key, run_batch


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key, **kwargs):
        body = self.objects[(Bucket, Key)]

        class Body:
            def read(self_inner):
                return body
        return {"Body": Body(), "VersionId": "v7"}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body


def golden_bytes(n=4):
    rows = [{"id": f"g{i}", "text": f"t{i}", "route": "cards", "lang": "en", "urgency": 0} for i in range(n)]
    return "\n".join(json.dumps(r) for r in rows).encode()


def test_load_dataset_reads_jsonl_and_reports_the_object_version():
    s3 = FakeS3({("router-eval", "golden/golden.jsonl"): golden_bytes()})
    rows, version = load_dataset(s3, "router-eval", "golden/golden.jsonl")
    assert len(rows) == 4 and version == "v7"


def test_report_key_is_stable_and_sortable():
    assert report_key("b0", "golden", "2026-11-11T10:00:00") == "reports/2026-11-11T10:00:00-b0-golden.json"


def test_run_batch_writes_a_report_that_cites_the_dataset_version():
    s3 = FakeS3({("router-eval", "golden/golden.jsonl"): golden_bytes()})

    class AlwaysCards:
        def predict(self, rows):
            from harness import Prediction
            return [Prediction(route="cards", confidence=0.9) for _ in rows]
    key = run_batch(s3, AlwaysCards(), router_name="b0", dataset_key="golden/golden.jsonl",
                    bucket="router-eval", now="2026-11-11T10:00:00")
    report = json.loads(s3.objects[("router-eval", key)])
    assert report["dataset_version"] == "v7" and report["rows"] == 4 and report["metrics"]["accuracy"] == 1.0
