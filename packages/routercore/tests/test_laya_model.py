import math

from routercore.adapters.laya_model import LayaModel, build_questions
from routercore.application.contracts import InboundMessage
from routercore.domain.calibration import Calibration

ROUTES = {"fees": "fees and exchange rates", "security": "lost or stolen cards", "general": "chit-chat"}


class FakeAgent:
    """Shaped like laya.Agent.predict_batch results."""

    def __init__(self):
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((states, questions, kwargs))
        answers = {"route": {"probabilities": {"fees": 0.8, "security": 0.15, "general": 0.05}},
                   "urgency": {"probabilities": {"0": 0.7, "1": 0.2, "2": 0.05, "3": 0.05}}}
        return [{"answers": answers} for _ in states]


def message(text="hello", lang=None):
    return InboundMessage(message_id="m", customer_id="c", channel="chat", text=text, lang=lang, ts_sent_ns=0)


def test_build_questions_exposes_routes_and_urgency_levels():
    q = build_questions(ROUTES)
    assert q["route"]["criteria"] == ROUTES
    assert set(q["urgency"]["criteria"]) == {"0", "1", "2", "3"}
    assert q["route"]["type"] == q["urgency"]["type"] == "choice"


def test_assess_maps_laya_answers_to_assessments():
    agent = FakeAgent()
    model = LayaModel(agent, ROUTES, model_version="laya-test")
    [a] = model.assess([message("what is the fee?", "en")])
    assert a.route_p["fees"] == 0.8 and a.urgency_p["3"] == 0.05
    assert a.model_version == "laya-test" and a.lang_supported is True
    assert agent.calls[0][0] == ["what is the fee?"]


def test_assess_applies_calibration_per_question_and_language():
    model = LayaModel(FakeAgent(), ROUTES, model_version="v",
                      calibration=Calibration({"intent": {"default": 1.0, "es": 3.0}}))
    [en] = model.assess([message(lang="en")])
    [es] = model.assess([message(lang="es")])
    assert math.isclose(en.route_p["fees"], 0.8) and es.route_p["fees"] < 0.8
    assert math.isclose(sum(es.route_p.values()), 1.0)


def test_unsupported_language_is_flagged():
    model = LayaModel(FakeAgent(), ROUTES, model_version="v", supported_langs={"en", "es"})
    batch = [message(lang="en"), message(lang="de"), message(lang=None)]
    flags = [a.lang_supported for a in model.assess(batch)]
    assert flags == [True, False, True]
