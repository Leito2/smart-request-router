"""API entrypoint: python -m services.api.main  (env: BROKER, MODEL, PORT)."""
import json
import os

import uvicorn
from confluent_kafka import Producer

from routercore.application.contracts import InboundMessage
from services.api.app import create_app
from services.composition import build_route_messages


def main() -> None:
    producer = Producer({"bootstrap.servers": os.environ.get("BROKER", "localhost:9092")})

    def publish(message: InboundMessage) -> None:
        producer.produce("inbound-messages", key=message.customer_id, value=json.dumps(message.model_dump()))
        producer.poll(0)

    uvicorn.run(create_app(build_route_messages(), publish), host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
