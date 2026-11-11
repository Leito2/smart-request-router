"""System 2 worker (PLAN §3.6): consumes `escalations`, runs the agent, publishes the decision to `routed`.

It lives outside the dataflow (ADR-1): an agent takes seconds and must not block System 1.
Concurrency 2, 20 s timeout per case; anything that fails goes to `triage`, nothing is lost.
Env: BROKER, GATEWAY_URL, GATEWAY_KEY, LLM_MODEL, REDIS_URL, P1_URL, P3_URL.
"""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import redis
from confluent_kafka import Consumer, Producer
from prometheus_client import Counter, Histogram, start_http_server

from routercore.adapters.route_map_config import load_route_map
from routercore.application.contracts import Escalation
from services.composition import ROOT
from services.dispatcher.adapters import HttpKnowledgeBase, HttpPaymentDecisions
from services.system2.adapters import GatewayLlm, LiveTools, RedisHistory
from services.system2.graph import FALLBACK_ROUTE, build_agent

log = logging.getLogger("system2")
CONCURRENCY, TIMEOUT_S = 2, 20
S2_LATENCY = Histogram("s2_latency_seconds", "System 2 time per escalation",
                       buckets=(.5, 1, 2, 5, 10, 20))
S2_FALLBACK = Counter("s2_fallback_total", "System 2 decisions that fell back to triage")
S2_TOKENS = Counter("s2_tokens_total", "LLM tokens spent by System 2")


def fallback(escalation: dict, error: str) -> dict:
    message = escalation["message"]
    return {"message_id": message["message_id"], "route": FALLBACK_ROUTE, "priority": "high",
            "decided_by": "system2", "confidence": 0.0, "model_version": "system2-fallback",
            "threshold_set": "s2", "ts_routed_ns": 0, "text": message["text"],
            "rationale": f"fallback: {error}", "evidence": []}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    start_http_server(int(os.environ.get("METRICS_PORT", "8003")))
    routes = load_route_map(ROOT / "config" / "route_map.yaml").routes
    llm = GatewayLlm(os.environ.get("GATEWAY_URL", "http://localhost:8080"),
                     model=os.environ.get("LLM_MODEL", "smart_router"), api_key=os.environ.get("GATEWAY_KEY"))
    tools = LiveTools(RedisHistory(redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))),
                      HttpPaymentDecisions(os.environ.get("P1_URL", "http://localhost:8010")),
                      HttpKnowledgeBase(os.environ.get("P3_URL", "http://localhost:8010")))
    agent = build_agent(llm, tools, routes)
    broker = os.environ.get("BROKER", "localhost:9092")
    consumer = Consumer({"bootstrap.servers": broker, "group.id": "router-system2",
                         "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    producer = Producer({"bootstrap.servers": broker})
    consumer.subscribe(["escalations"])
    pool = ThreadPoolExecutor(max_workers=CONCURRENCY)

    def resolve(raw: dict) -> dict:
        Escalation.model_validate(raw)
        return agent.invoke({"escalation": raw})["routed"]

    while True:
        message = consumer.poll(1.0)
        if message is None or message.error():
            continue
        raw = json.loads(message.value())
        began, tokens_before = time.perf_counter(), llm.tokens
        future = pool.submit(resolve, raw)
        try:
            decision = future.result(timeout=TIMEOUT_S)
        except (FutureTimeout, Exception) as error:        # timeout, bad LLM, down gateway: human triage
            decision = fallback(raw, type(error).__name__)
        S2_LATENCY.observe(time.perf_counter() - began)
        S2_TOKENS.inc(llm.tokens - tokens_before)
        if decision["route"] == FALLBACK_ROUTE:
            S2_FALLBACK.inc()
        producer.produce("routed", key=decision["message_id"], value=json.dumps(decision))
        producer.flush()
        consumer.commit(message)
        log.info("%s -> %s (%s)", decision["message_id"], decision["route"],
                 decision.get("rationale", "")[:60])


if __name__ == "__main__":
    main()
