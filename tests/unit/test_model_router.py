from routers import ModelRouter

from routercore.adapters.rules_model import RulesModel
from routercore.domain.policy import Policy
from routercore.domain.routes import RouteMap

ROUTE_MAP = RouteMap(routes={r: r for r in ("security", "cards", "general")}, intents={})
ROWS = [
    {"id": "1", "text": "hello there"},
    {"id": "2", "text": "my card was stolen"},
]


def test_without_a_policy_every_message_is_answered_by_the_model():
    predictions = ModelRouter(RulesModel(ROUTE_MAP), policy=None).predict(ROWS)

    assert [p.route for p in predictions] == ["general", "security"]
    assert [p.urgency for p in predictions] == [0, 3]
    assert not any(p.escalated for p in predictions)
    assert all(p.llm_calls == 0 for p in predictions)


def test_with_a_policy_the_urgent_message_escalates_but_keeps_the_models_best_guess():
    predictions = ModelRouter(RulesModel(ROUTE_MAP), policy=Policy()).predict(ROWS)

    assert not predictions[0].escalated and predictions[0].route == "general"
    assert predictions[1].escalated and predictions[1].route == "security"


def test_llm_answers_are_parsed_leniently_and_invalid_ones_fall_back_to_general():
    from routers import parse_answer

    routes = {"security", "cards", "general"}

    assert parse_answer('Sure! {"route": "security", "urgency": 3}', routes) == ("security", 3)
    assert parse_answer('{"route": "security", "urgency": 9}', routes) == ("security", 3)
    assert parse_answer('{"route": "pizza", "urgency": 1}', routes) == ("general", 1)
    assert parse_answer("I think it is about cards", routes) == ("general", 0)
