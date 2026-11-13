"""Surge detector (PLAN §3.9): counts decisions per route in 1-minute windows and emits `alerts`.

Reads `routed` and `escalations` (event time = Kafka timestamp), keeps one EWMA baseline per route in memory
(a restart means a new warm-up) and writes an alert when z > threshold for N consecutive windows.
Run: python -m services.dataflow.surge_app   (env: BROKER, METRICS_PORT)
"""
import os
from pathlib import Path

from prometheus_client import Counter, start_http_server
from quixstreams import Application

from routercore.adapters.surge_config import load_surge_config
from routercore.domain.surge import SurgeMonitor
from services.composition import ROOT

DECISIONS = Counter("router_route_messages_total", "Decisions by route", ["route"])
ALERTS = Counter("router_surge_alerts_total", "Surge alerts by route", ["route"])


def route_of(value: dict) -> str:
    if "route" in value:
        return value["route"]
    intent = value["laya"]["intent"]                 # an escalation: the route System 1 leaned to
    return max(intent, key=intent.__getitem__)


def main() -> None:
    config_path = Path(os.environ.get("SURGE_CONFIG", ROOT / "config" / "surge.yaml"))
    monitor = SurgeMonitor(load_surge_config(config_path))
    start_http_server(int(os.environ.get("METRICS_PORT", "8002")))
    app = Application(broker_address=os.environ.get("BROKER", "localhost:9092"),
                      consumer_group="router-surge", auto_offset_reset="earliest",
                      commit_interval=float(os.environ.get("COMMIT_INTERVAL_S", "0.5")))  # Quix flushes output per commit (default 5 s)
    alerts = app.topic("alerts", key_serializer="str", value_serializer="json")

    def observe(value: dict, key, timestamp_ms: int, headers) -> list[dict]:
        route = route_of(value)
        DECISIONS.labels(route).inc()
        fired = monitor.add(route, timestamp_ms * 1_000_000)
        if fired:
            ALERTS.labels(route).inc(len(fired))
        return [{"intent": a.intent, "window": a.window, "volume": a.volume, "baseline": a.baseline,
                 "zscore": a.zscore, "examples": [value.get("text") or value.get("message", {}).get("text")]}
                for a in fired]

    for name in ("routed", "escalations"):
        topic = app.topic(name, key_deserializer="str", value_deserializer="json")
        sdf = app.dataframe(topic).apply(observe, metadata=True, expand=True)
        sdf.to_topic(alerts, key=lambda alert: alert["intent"])
    app.run()


if __name__ == "__main__":
    main()
