"""Dispatcher (PLAN §3.8): consumes `routed`, executes each decision once (idempotent by message_id).

At-least-once consumption (offset committed after the dispatch) + the Postgres unique key = effectively once.
Env: BROKER, REDIS_URL, DATABASE_URL, P1_URL, P3_URL (default: the mocks on :8010).
"""
import json
import logging
import os

import psycopg
import redis
from confluent_kafka import Consumer

from routercore.adapters.http_clients import HttpKnowledgeBase, HttpPaymentDecisions
from routercore.application.contracts import Routed
from routercore.application.dispatch import Dispatch
from services.dispatcher.adapters import PostgresDecisionLog, RedisTeamQueues

log = logging.getLogger("dispatcher")


def build_dispatch() -> Dispatch:
    connection = psycopg.connect(os.environ.get("DATABASE_URL", "postgresql://router:router@localhost:5432/router"))
    queues = RedisTeamQueues(redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
    return Dispatch(PostgresDecisionLog(connection), queues,
                    HttpKnowledgeBase(os.environ.get("P3_URL", "http://localhost:8010")),
                    HttpPaymentDecisions(os.environ.get("P1_URL", "http://localhost:8010")))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dispatch = build_dispatch()
    consumer = Consumer({"bootstrap.servers": os.environ.get("BROKER", "localhost:9092"),
                         "group.id": "router-dispatcher", "enable.auto.commit": False,
                         "auto.offset.reset": "earliest"})
    consumer.subscribe(["routed"])
    while True:
        message = consumer.poll(1.0)
        if message is None:
            continue
        if message.error():
            log.error("consumer error: %s", message.error())
            continue
        outcome = dispatch(Routed.model_validate(json.loads(message.value())))
        log.info("%s -> %s", message.key().decode(), outcome)
        consumer.commit(message)         # only after the effect: a crash in between means a redelivery


if __name__ == "__main__":
    main()
