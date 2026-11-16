"""Outbound adapters only the dispatcher uses: Postgres decision log and Redis Streams queues."""
from routercore.application.contracts import Routed


class PostgresDecisionLog:
    """`record` is the idempotency gate (ADR-8): the primary key rejects a message_id seen before."""

    SQL = ("INSERT INTO decisions (message_id, route, priority, decided_by, confidence, model_version, "
           "threshold_set) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (message_id) DO NOTHING")

    def __init__(self, connection):
        self._connection = connection

    def record(self, routed: Routed) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(self.SQL, (routed.message_id, routed.route, routed.priority, routed.decided_by,
                                      routed.confidence, routed.model_version, routed.threshold_set))
            inserted = cursor.rowcount == 1
        self._connection.commit()
        return inserted


class RedisTeamQueues:
    def __init__(self, redis_client):
        self._redis = redis_client

    def enqueue(self, team: str, routed: Routed) -> None:
        self._redis.xadd(f"queue:{team}", {"message_id": routed.message_id, "priority": routed.priority,
                                           "text": routed.text or ""})
