"""Prompt-attack screening of escalated messages with Llama Prompt Guard 2 86M, in `monitor` mode.

The System 2 agent reads customer text and calls tools, so it is the part of the router a prompt injection can
reach. The screen runs before the agent and only observes: a flagged message is counted
(`s2_prompt_attack_flagged_total`) and its score goes to the Langfuse trace, but the agent still handles it.
Measured on the router's data (`eval/prompt_guard_eval.py`, `docs/results/prompt_guard.json`): at
threshold 0.5 it flags 1 of 4,202 real customer messages and catches 32.5 % of a hand-written banking
attack set, so it is a signal for review, not a defence; blocking (`enforce`) would need far better recall.

Env: PROMPT_GUARD_MODE (`monitor` default, `off`), PROMPT_GUARD_DIR (default models/prompt_guard_86m,
the gated meta-llama/Llama-Prompt-Guard-2-86M weights), PROMPT_GUARD_THRESHOLD (default 0.5). Without
the weights, or without torch/transformers, the screen is off and System 2 runs as before.
"""
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger("system2.prompt_guard")
DEFAULT_THRESHOLD = 0.5


class Scorer(Protocol):
    def __call__(self, text: str) -> float: ...


@dataclass(frozen=True)
class Screening:
    score: float | None          # probability of a prompt attack; None when the model gave no verdict
    flagged: bool
    seconds: float

    def as_metadata(self) -> dict:
        return {"prompt_attack_score": None if self.score is None else round(self.score, 4),
                "prompt_attack_flagged": self.flagged}


class PromptGuardModel:
    """Llama Prompt Guard 2 on CPU through transformers (512-token window; longer text is truncated)."""

    def __init__(self, directory: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
        labels = {name.lower(): int(index) for index, name in self._model.config.id2label.items()}
        self._positive = labels.get("malicious", labels.get("label_1", 1))

    def __call__(self, text: str) -> float:
        with self._torch.inference_mode():
            encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            return float(self._torch.softmax(self._model(**encoded).logits, dim=-1)[0, self._positive])


class PromptGuardMonitor:
    """Scores a message and reports a flag; never raises and never changes the routing."""

    def __init__(self, scorer: Scorer, *, threshold: float = DEFAULT_THRESHOLD,
                 on_flag: Callable[[str | None], None] = lambda lang: None,
                 on_screen: Callable[[float], None] = lambda seconds: None):
        self._scorer, self._threshold = scorer, threshold
        self._on_flag, self._on_screen = on_flag, on_screen

    def screen(self, text: str, lang: str | None = None) -> Screening:
        began = time.perf_counter()
        try:
            score = self._scorer(text)
        except Exception as error:              # a broken screen must never stop routing
            log.warning("prompt guard gave no verdict: %s", type(error).__name__)
            score = None
        seconds = time.perf_counter() - began
        flagged = score is not None and score >= self._threshold
        self._on_screen(seconds)
        if flagged:
            self._on_flag(lang)
        return Screening(score, flagged, seconds)


def from_env(root: Path, **hooks) -> PromptGuardMonitor | None:
    if os.environ.get("PROMPT_GUARD_MODE", "monitor").lower() == "off":
        return None
    directory = Path(os.environ.get("PROMPT_GUARD_DIR", root / "models" / "prompt_guard_86m"))
    if not (directory / "config.json").exists():
        log.info("prompt guard off: no model in %s", directory)
        return None
    try:
        model = PromptGuardModel(directory)
    except Exception as error:                  # torch/transformers missing, or bad weights
        log.warning("prompt guard off: %s: %s", type(error).__name__, error)
        return None
    threshold = float(os.environ.get("PROMPT_GUARD_THRESHOLD", DEFAULT_THRESHOLD))
    log.info("prompt guard on (monitor, threshold %.2f): %s", threshold, directory)
    return PromptGuardMonitor(model, threshold=threshold, **hooks)
