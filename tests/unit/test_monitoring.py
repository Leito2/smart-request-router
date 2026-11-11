from drift_monitor import confidence_bucket, feature_counts, reference_counts
from feedback import merge_feedback


def test_confidence_bucket_edges():
    assert confidence_bucket(0.99) == "0.9-1.0" and confidence_bucket(0.05) == "0.0-0.5"
    assert confidence_bucket(0.75) == "0.7-0.8" and confidence_bucket(1.0) == "0.9-1.0"


def test_feature_counts_from_decisions():
    decisions = [{"route": "cards", "lang": "en", "confidence": 0.95, "kind": "routed"},
                 {"route": "cards", "lang": "es", "confidence": 0.6, "kind": "routed"},
                 {"route": "fees", "lang": "es", "confidence": None, "kind": "escalation"}]
    counts = feature_counts(decisions)
    assert counts["route"] == {"cards": 2, "fees": 1}
    assert counts["lang"] == {"en": 1, "es": 2}
    assert counts["outcome"] == {"routed": 2, "escalation": 1}
    assert counts["confidence"] == {"0.9-1.0": 1, "0.6-0.7": 1}


def test_reference_counts_use_the_train_split_only():
    rows = [{"split": "train", "route": "a", "lang": "en"}, {"split": "test", "route": "b", "lang": "en"}]
    assert reference_counts(rows)["route"] == {"a": 1}


def test_merge_feedback_adds_corrected_examples_to_train_without_duplicates():
    train = [{"id": "x1", "text": "hello", "route": "general", "urgency": 0, "lang": "en", "split": "train"}]
    feedback = [{"message_id": "m1", "text": "my card was stolen", "correct_route": "security", "lang": "en"},
                {"message_id": "m2", "text": "hello", "correct_route": "general", "lang": "en"}]
    merged = merge_feedback(train, feedback)
    assert [r["text"] for r in merged] == ["hello", "my card was stolen"]
    new = merged[1]
    assert new["route"] == "security" and new["split"] == "train" and new["id"] == "fb-m1"
