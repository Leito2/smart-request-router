import pytest

np = pytest.importorskip("numpy")

from routercore.adapters.setfit_model import IntentUrgencyHead, clauses, intent_label  # noqa: E402


class FakeEncoder:
    """One-hot over two themes: [calm, stolen card]."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([[0.0, 1.0] if "stole" in t else [1.0, 0.0] for t in texts])


class FakeClassifier:
    classes_ = np.array(["card_arrival", "lost_or_stolen_card", "topping_up"])

    def predict_proba(self, embeddings):
        # calm text: topping_up (urgency 0) 0.9, card_arrival (urgency 1) 0.1; stolen: lost_or_stolen_card
        return np.array([[0.0, 1.0, 0.0] if e[1] else [0.1, 0.0, 0.9] for e in embeddings])


def head():
    return IntentUrgencyHead(encoder=FakeEncoder(), classifier=FakeClassifier(), encoder_name="fake",
                             urgency_of_class={"card_arrival": 1, "lost_or_stolen_card": 3, "topping_up": 0})


def test_clauses_splits_sentences_and_ignores_single_sentence_and_tiny_fragments():
    assert clauses("Why is my top up pending? Also, someone stole my card.") == [
        "Why is my top up pending?", "Also, someone stole my card."]
    assert clauses("Just one sentence here") == []
    assert clauses("Thanks. Ok.") == []


def test_intent_label_keeps_one_class_per_urgency_level_for_rows_without_an_intent():
    assert intent_label({"intent": "lost_or_stolen_card", "urgency": 3}) == "lost_or_stolen_card"
    assert intent_label({"intent": None, "urgency": 2}) == "none_u2"


def test_probabilities_of_the_intents_are_folded_into_urgency_levels():
    calm, stolen = head().predict_proba(["when does my top up arrive", "someone stole my card"])
    assert calm.round(2).tolist() == [0.9, 0.1, 0.0, 0.0]
    assert stolen.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_an_urgent_second_sentence_makes_the_whole_message_urgent():
    text = "Why is my top up still pending? On top of that, someone stole my card."
    (row,) = head().predict_proba([text])
    assert row[3] == 1.0                      # as a whole it reads as calm text, the clause does not
