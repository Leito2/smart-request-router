"""LLM-as-a-Judge building blocks (PLAN §12): rubric, pairwise (position-swapped), claims, panel.

A judge only needs `complete(system, user) -> text` (async), so any gateway alias or local model plugs in.
"""
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Complete = Callable[[str, str], Awaitable[str]]


def parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


@dataclass(frozen=True)
class RubricResult:
    score: float | None
    reason: str = ""


class RubricJudge:
    """Scores an item against a written rubric. An unparseable answer gives score None, never an exception."""

    def __init__(self, complete: Complete, rubric: str, low: int = 1, high: int = 5):
        self._complete, self._rubric, self._low, self._high = complete, rubric, low, high

    async def judge(self, item: dict[str, Any]) -> RubricResult:
        system = (f"You are a strict evaluator. Rubric: {self._rubric}\nScore from {self._low} (worst) to "
                  f'{self._high} (best). Reply JSON only: {{"score": <int>, "reason": "<one sentence>"}}.')
        answer = parse_json(await self._complete(system, json.dumps(item, ensure_ascii=False)))
        score = answer.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            return RubricResult(None)
        return RubricResult(min(max(score, self._low), self._high), str(answer.get("reason", "")))


class PairwiseJudge:
    """Which of two answers is better? Asked in both orders; a verdict that flips with the order is a tie."""

    def __init__(self, complete: Complete, criterion: str = "the answer that is more correct and useful"):
        self._complete, self._criterion = complete, criterion

    async def _ask(self, question: str, first: str, second: str) -> str | None:
        system = f'Pick {self._criterion}. Reply JSON only: {{"winner": "A" or "B" or "tie"}}.'
        user = json.dumps({"question": question, "A": first, "B": second}, ensure_ascii=False)
        return parse_json(await self._complete(system, user)).get("winner")

    async def judge(self, question: str, answer_1: str, answer_2: str) -> str:
        forward, backward = await asyncio.gather(self._ask(question, answer_1, answer_2),
                                                 self._ask(question, answer_2, answer_1))
        if forward == "A" and backward == "B":
            return answer_1
        if forward == "B" and backward == "A":
            return answer_2
        return "tie"


@dataclass(frozen=True)
class ClaimsResult:
    score: float
    unsupported: list[str] = field(default_factory=list)


class ClaimsJudge:
    """Faithfulness: split the answer into claims and check each against the evidence."""

    def __init__(self, complete: Complete):
        self._complete = complete

    async def judge(self, answer: str, evidence: str) -> ClaimsResult:
        system = ("List the factual claims of the answer and mark whether the evidence supports each. "
                  'Reply JSON only: {"claims": [{"claim": "...", "supported": true or false}]}.')
        user = json.dumps({"answer": answer, "evidence": evidence}, ensure_ascii=False)
        claims = parse_json(await self._complete(system, user)).get("claims", [])
        claims = [c for c in claims if isinstance(c, dict) and "supported" in c]
        if not claims:
            return ClaimsResult(0.0)
        unsupported = [str(c.get("claim", "")) for c in claims if not c["supported"]]
        return ClaimsResult(1 - len(unsupported) / len(claims), unsupported)


@dataclass(frozen=True)
class PanelResult:
    score: float | None
    scores: list[float]
    disagreement: bool


class Panel:
    """Several rubric judges (e.g. local Gemma + the golden judge): mean score, flagged when they disagree."""

    def __init__(self, judges: Sequence[RubricJudge], max_spread: float = 1.0):
        self._judges, self._max_spread = judges, max_spread

    async def judge(self, item: dict[str, Any]) -> PanelResult:
        results = await asyncio.gather(*(j.judge(item) for j in self._judges))
        scores = [r.score for r in results if r.score is not None]
        if not scores:
            return PanelResult(None, [], True)
        return PanelResult(sum(scores) / len(scores), scores, max(scores) - min(scores) > self._max_spread)
