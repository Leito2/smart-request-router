"""Quix Streams dataflow (PLAN §3.3): inbound-messages → routed | escalations | dlq.

Run: python -m services.dataflow.main   (env: BROKER, MODEL, METRICS_PORT). One model load per process.
Quix applies operators per message; Laya is called with a micro-batch of one here (ADR-3 trade-off is measured
in the M3 batch benchmark and discussed in the README).
"""
import os
import time

from prometheus_client import Counter, Histogram, start_http_server
from quixstreams import Application

from services.composition import build_route_messages
from services.dataflow.pipeline import process

OUT_TOPICS = ("routed", "escalations", "dlq")
KEY_OF = {"routed": lambda v: v["message_id"], "escalations": lambda v: v["message"]["message_id"],
          "dlq": lambda v: str(v["raw"].get("message_id", ""))}
OUTCOMES = Counter("router_outcomes_total", "Messages by output topic", ["topic"])
ESCALATIONS = Counter("router_escalations_total", "Escalations by reason", ["reason"])
CONFIDENCE = Histogram("router_confidence", "System 1 confidence of routed messages",
                       buckets=(.5, .6, .7, .8, .85, .9, .95, .99, 1.0))
LATENCY = Histogram("router_process_seconds", "Per-message processing time",
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))


def main() -> None:
    route = build_route_messages()
    start_http_server(int(os.environ.get("METRICS_PORT", "8001")))
    app = Application(broker_address=os.environ.get("BROKER", "localhost:9092"),
                      consumer_group="router-dataflow", auto_offset_reset="earliest",
                      state_dir=os.environ.get("STATE_DIR", ".quix-state"),
                      commit_interval=float(os.environ.get("COMMIT_INTERVAL_S", "0.5")))  # Quix flushes output per commit (default 5 s)
    inbound = app.topic("inbound-messages", key_deserializer="str", value_deserializer="json")
    outputs = {name: app.topic(name, key_serializer="str", value_serializer="json") for name in OUT_TOPICS}

    def step(value: dict, state) -> list[dict]:
        started = time.perf_counter()
        customer_state = {"recent": state.get("recent", [])}
        rows = process(value, customer_state, route)
        state.set("recent", customer_state["recent"])
        LATENCY.observe(time.perf_counter() - started)
        for topic, _, body in rows:
            if topic == "escalations":
                ESCALATIONS.labels(body["reason"]).inc()
            elif topic == "routed":
                CONFIDENCE.observe(body["confidence"])
        return [{"topic": topic, "value": body} for topic, _, body in rows]

    sdf = app.dataframe(inbound).apply(step, stateful=True, expand=True)
    for name in OUT_TOPICS:
        branch = sdf[sdf.apply(lambda row, n=name: row["topic"] == n)]
        branch = branch.apply(lambda row, n=name: (OUTCOMES.labels(n).inc(), row["value"])[1])
        branch.to_topic(outputs[name], key=KEY_OF[name])
    app.run()


if __name__ == "__main__":
    main()
