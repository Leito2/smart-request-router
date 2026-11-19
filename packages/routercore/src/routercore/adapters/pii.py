"""PII detection and redaction with Presidio (PLAN §12 data governance).

Pattern recognizers only (card with Luhn, IBAN, email, phone, IP, Spanish NIF/NIE, US SSN, crypto
wallet), on a blank spaCy pipeline: no model download, no NER, so names and addresses are NOT detected.
That is the honest scope: in support text the dangerous identifiers are numbers and emails.

Text is analysed as English and as Spanish and the spans are merged, because the NIF/NIE recognizers are
Spanish-only and the rest are language-independent. `PiiRedactor.mask` is shaped for Langfuse's `mask=` hook:
it walks dicts and lists and redacts every string. Heavy deps are imported lazily (extra `pii`).
"""
from dataclasses import dataclass
from typing import Any

ENTITIES = ("CREDIT_CARD", "IBAN_CODE", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "ES_NIF", "ES_NIE",
            "US_SSN", "CRYPTO")
LANGUAGES = ("en", "es")
MIN_SCORE = 0.3        # PHONE_NUMBER scores 0.4 even for well-formed numbers; lower than this is noise
STRONG = 0.8           # at or above this a finding is a hard hit (card, IBAN, email, NIF all score 1.0)


@dataclass(frozen=True)
class Finding:
    entity: str
    start: int
    end: int
    score: float


class PiiRedactor:
    def __init__(self, *, entities: tuple[str, ...] = ENTITIES, min_score: float = MIN_SCORE):
        import spacy
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        class BlankEngine(SpacyNlpEngine):
            def __init__(self):
                super().__init__(models=[{"lang_code": lang, "model_name": "blank"} for lang in LANGUAGES])
                self.nlp = {lang: spacy.blank(lang) for lang in LANGUAGES}

        registry = RecognizerRegistry(supported_languages=list(LANGUAGES))
        registry.load_predefined_recognizers(languages=list(LANGUAGES), nlp_engine=BlankEngine())
        self._analyzer = AnalyzerEngine(nlp_engine=BlankEngine(), registry=registry,
                                        supported_languages=list(LANGUAGES))
        self._entities, self._min_score = list(entities), min_score

    def findings(self, text: str) -> list[Finding]:
        """Merged, non-overlapping spans found in either language (the highest score wins an overlap)."""
        found = [Finding(r.entity_type, r.start, r.end, r.score)
                 for lang in LANGUAGES
                 for r in self._analyzer.analyze(text=text, language=lang, entities=self._entities,
                                                 score_threshold=self._min_score)]
        found.sort(key=lambda f: (-f.score, f.start))
        kept: list[Finding] = []
        for f in found:
            if all(f.end <= k.start or f.start >= k.end for k in kept):
                kept.append(f)
        return sorted(kept, key=lambda f: f.start)

    def redact(self, text: str) -> str:
        out, last = [], 0
        for f in self.findings(text):
            out += [text[last:f.start], f"<{f.entity}>"]
            last = f.end
        return "".join(out) + text[last:]

    def mask(self, *, data: Any, **_: Any) -> Any:
        """Langfuse `mask=` hook: redact every string inside dicts, lists and tuples."""
        if isinstance(data, str):
            return self.redact(data)
        if isinstance(data, dict):
            return {k: self.mask(data=v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self.mask(data=v) for v in data]
        return data
