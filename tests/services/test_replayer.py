from services.replayer.main import build_events, pick_rows


def rows(n_en=6, n_es=4):
    return ([{"id": f"e{i}", "text": f"en {i}", "lang": "en"} for i in range(n_en)]
            + [{"id": f"s{i}", "text": f"es {i}", "lang": "es"} for i in range(n_es)])


def test_lang_mix_is_respected():
    picked = pick_rows(rows(), 10, {"es": 0.6, "en": 0.4}, seed=0)
    assert sum(r["lang"] == "es" for r in picked) == 6 and len(picked) == 10


def test_events_follow_the_inbound_contract_and_are_keyed_by_customer():
    events = build_events(rows(), customers=3, start_ns=100, seed=0)
    assert len(events) == 10
    assert all(e["customer_id"].startswith("c_") and e["channel"] in {"chat", "email", "app"} for e in events)
    assert events[0]["text"] == "en 0" and events[0]["lang"] == "en"
    assert len({e["message_id"] for e in events}) == 10
