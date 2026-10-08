from routercore.domain.policy import decide

CALM = {"0": 0.7, "1": 0.25, "2": 0.04, "3": 0.01}


def test_confident_message_stays_in_system1():
    assert decide({"disputes": 0.93, "cards": 0.04, "general": 0.03}, CALM, True) == ("disputes", "system1")


def test_urgency_safety_net_wins_even_when_route_is_confident():
    urgent = {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1}
    assert decide({"security": 0.95, "general": 0.05}, urgent, True) == ("system2", "urgency_safety_net")


def test_close_top_two_is_multi_intent():
    assert decide({"disputes": 0.48, "cards": 0.44, "general": 0.08}, CALM, True)[1] == "multi_intent"


def test_unsupported_language_is_ood():
    assert decide({"disputes": 0.99, "general": 0.01}, CALM, False)[1] == "ood"
