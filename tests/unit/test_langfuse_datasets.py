from types import SimpleNamespace

from scripts.langfuse_datasets import prune, to_item, upload

ROW = {"id": "hc-0:en", "split": "hard", "text": "hello", "lang": "en", "route": "general", "urgency": 0,
       "category": "general", "should_escalate": False, "intent": None, "ignored": "x"}


class FakeItems:
    def __init__(self, existing):
        self.existing, self.deleted = existing, []

    def list(self, *, dataset_name, page, limit):
        data = [SimpleNamespace(id=i) for i in self.existing]
        return SimpleNamespace(data=data, meta=SimpleNamespace(total_pages=1))

    def delete(self, item_id):
        self.deleted.append(item_id)


class FakeClient:
    def __init__(self, existing=()):
        self.api = SimpleNamespace(dataset_items=FakeItems(list(existing)))
        self.created, self.datasets = [], []

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs["name"])

    def create_dataset_item(self, **kwargs):
        self.created.append(kwargs)

    def flush(self):
        pass


def test_item_ids_are_namespaced_per_dataset_so_shared_rows_do_not_collide():
    golden, hard = to_item("router-golden-v1", ROW), to_item("router-hard-v1", ROW)
    assert golden["id"] != hard["id"] and hard["id"] == "router-hard-v1/hc-0:en"


def test_item_carries_input_expected_output_and_metadata_only():
    item = to_item("router-hard-v1", ROW)
    assert item["input"] == {"text": "hello", "lang": "en"}
    assert item["expected_output"] == {"route": "general", "urgency": 0}
    expected = {"set": "hard", "intent": None, "category": "general", "should_escalate": False}
    assert item["metadata"] == expected


def test_prune_deletes_only_items_not_kept():
    client = FakeClient(existing=["old-id", "router-hard-v1/hc-0:en"])
    assert prune(client, "router-hard-v1", {"router-hard-v1/hc-0:en"}) == 1
    assert client.api.dataset_items.deleted == ["old-id"]


def test_upload_creates_dataset_and_items_and_reports_removed():
    client = FakeClient(existing=["hc-0:en"])          # legacy id scheme
    removed = upload(client, "router-hard-v1", [ROW], "desc")
    assert removed == 1 and client.datasets == ["router-hard-v1"]
    assert [c["id"] for c in client.created] == ["router-hard-v1/hc-0:en"]
