"""Deterministic token-matching metrics (no model needed): the cheap first layer before any LLM judge."""
import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the|el|la|los|las|un|una|unos|unas)\b")
_CITATION = re.compile(r"\[(\d+)\]")


def normalize(text: str) -> str:
    """Lowercase, strip accents, punctuation, articles (EN/ES) and extra whitespace (SQuAD-style)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 over normalized bags of tokens."""
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def citation_ids(text: str) -> set[int]:
    """Citation markers like [1] [3] found in an answer."""
    return {int(n) for n in _CITATION.findall(text)}


def citation_precision(answer: str, valid_ids: set[int]) -> float:
    """Share of cited ids that point to a retrieved source (1.0 when nothing is cited)."""
    cited = citation_ids(answer)
    if not cited:
        return 1.0
    return len(cited & valid_ids) / len(cited)
