from baselines.rules import RulesModel

from routercore.application.contracts import InboundMessage
from routercore.domain.routes import RouteMap

ROUTE_MAP = RouteMap(
    routes={r: r for r in ("security", "cards", "topup", "fees", "general")}, intents={})


def msg(text: str) -> InboundMessage:
    return InboundMessage(message_id="m", customer_id="c", channel="chat", text=text, ts_sent_ns=0)


def top(p: dict[str, float]) -> str:
    return max(p, key=p.__getitem__)


def test_rules_baseline_answers_with_normalised_distributions_over_all_routes():
    model = RulesModel(ROUTE_MAP)

    stolen, hello, pin = model.assess([msg("my card was stolen"), msg("hello there"), msg("change my PIN")])

    for assessment in (stolen, hello, pin):
        assert abs(sum(assessment.route_p.values()) - 1) < 1e-9
        assert set(assessment.route_p) == set(ROUTE_MAP.routes)
        assert abs(sum(assessment.urgency_p.values()) - 1) < 1e-9
        assert set(assessment.urgency_p) == {"0", "1", "2", "3"}
    assert (top(stolen.route_p), top(stolen.urgency_p)) == ("security", "3")
    assert (top(hello.route_p), top(hello.urgency_p)) == ("general", "0")
    assert top(pin.route_p) == "cards"


def test_rules_baseline_is_fooled_by_negation_which_is_why_it_is_the_floor():
    [negated] = RulesModel(ROUTE_MAP).assess([msg("my card was not stolen, I just want to change my PIN")])

    assert top(negated.route_p) == "security"     # the right answer is "cards": keywords cannot read "not"
