from services.replayer.main import build_events, parse_surge, pick_rows, surge_rows


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


def test_parse_surge_spec():
    assert parse_surge("card_declined:5:120:180") == ("card_declined", 5.0, 120.0, 180.0)


def test_surge_rows_add_extra_messages_of_one_route_inside_the_window():
    pool = [{"id": "a", "text": "x", "lang": "en", "route": "payments"},
            {"id": "b", "text": "y", "lang": "en", "route": "fees"}]
    extra = surge_rows(pool, "payments", factor=5, base_rate=10, at_s=120, duration_s=60, seed=0)
    assert len(extra) == int(10 * (5 - 1) * 60)            # (factor - 1) x base rate over the window
    assert all(row["route"] == "payments" for _, row in extra)
    assert all(120 <= t < 180 for t, _ in extra)
