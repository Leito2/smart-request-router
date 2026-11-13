"""Loads `.env` (git-ignored) into os.environ for the services that read secrets: gateway key, Langfuse keys.

Stdlib on purpose. Variables already set in the real environment win over the file, and empty values
count as unset (so a blank LANGFUSE_PUBLIC_KEY keeps tracing a no-op).
"""
import os
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip()
        if value:
            os.environ.setdefault(key.strip(), value)
