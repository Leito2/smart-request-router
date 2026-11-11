"""Automatic promotion with rollback (PLAN §4.7): a candidate that passes the gate becomes `production`; the
model it replaces is kept as `previous`, so a bad release is one command to undo.

Usage: python training/promote.py promote --version 4 --gate-passed | rollback   (MLflow registry aliases)
"""
import argparse
from typing import Protocol

MODEL_NAME = "laya-router"


class Registry(Protocol):
    def get_alias(self, name: str, alias: str) -> str | None: ...
    def set_alias(self, name: str, alias: str, version: str) -> None: ...


class MlflowRegistry:
    def __init__(self, tracking_uri: str | None = None):
        from mlflow import MlflowClient

        self._client = MlflowClient(tracking_uri=tracking_uri)

    def get_alias(self, name: str, alias: str) -> str | None:
        try:
            return self._client.get_model_version_by_alias(name, alias).version
        except Exception:
            return None

    def set_alias(self, name: str, alias: str, version: str) -> None:
        self._client.set_registered_model_alias(name, alias, version)


def promote(registry: Registry, name: str, version: str, *, gate_passed: bool) -> str:
    if not gate_passed:
        return "rejected"
    current = registry.get_alias(name, "production")
    if current is not None:
        registry.set_alias(name, "previous", current)
    registry.set_alias(name, "production", version)
    return "promoted"


def rollback(registry: Registry, name: str) -> str:
    previous = registry.get_alias(name, "previous")
    if previous is None:
        return "nothing_to_roll_back"
    registry.set_alias(name, "production", previous)
    return "rolled_back"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["promote", "rollback"])
    parser.add_argument("--version")
    parser.add_argument("--gate-passed", action="store_true")
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()
    registry = MlflowRegistry(args.tracking_uri)
    if args.action == "promote":
        print(promote(registry, MODEL_NAME, args.version, gate_passed=args.gate_passed))
    else:
        print(rollback(registry, MODEL_NAME))


if __name__ == "__main__":
    main()
