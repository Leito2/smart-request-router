from laya_finetune import to_laya_rows

ROUTES = {"fees": "fees", "general": "chit-chat"}


def test_rows_carry_both_questions_and_gold_labels():
    records = [{"text": "what is the fee?", "route": "fees", "urgency": 0, "split": "train"},
               {"text": "hola", "route": "general", "urgency": 0, "split": "val"}]
    [row] = to_laya_rows(records, ROUTES, split="train")
    assert row["state"] == "what is the fee?"
    assert row["expected"] == {"route": "fees", "urgency": "0"}
    assert set(row["questions"]) == {"route", "urgency"}
    assert row["questions"]["route"]["criteria"] == ROUTES
