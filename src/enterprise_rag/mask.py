"""PII 마스커: 실명(NER), 이메일·전화번호(정규식)를 찾아 가명 토큰으로 바꾼다. 색인 전에 쓴다."""

from functools import cache
from typing import Any, NamedTuple

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

EMAIL = r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
# 북미 번호: 415-555-0134, (415) 555-0134, 415.555.0134, +1 415 555 0134
PHONE = r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"


class PiiType(NamedTuple):
    entity: str  # Presidio 개체 이름
    token: str  # 가명 토큰


# config의 PII 종류 목록에 쓰는 이름 → PII 종류
TYPES = {
    "NAME": PiiType("PERSON", "[NAME]"),
    "EMAIL": PiiType("EMAIL", "[EMAIL]"),
    "PHONE": PiiType("PHONE", "[PHONE]"),
}


class Masker:
    def __init__(self, spacy_model: str, types: list[str]):
        unknown = [t for t in types if t not in TYPES]
        if unknown:
            raise ValueError(f"알 수 없는 PII 종류: {unknown} (쓸 수 있는 종류: {list(TYPES)})")
        self._analyzer = _analyzer(spacy_model)
        self._anonymizer = AnonymizerEngine()
        # PII 종류 목록에 있는 것만 찾는다. 목록이 비면 아무것도 바꾸지 않는다.
        self._entities = [TYPES[t].entity for t in types]
        self._operators = {
            t.entity: OperatorConfig("replace", {"new_value": t.token}) for t in TYPES.values()
        }

    def mask(self, text: str) -> str:
        if not self._entities:
            return text
        found = self._analyzer.analyze(text, language="en", entities=self._entities)
        return self._anonymizer.anonymize(text, found, self._operators).text

    def find(self, text: str) -> list[tuple[str, str]]:
        """`mask`가 가리는 PII를 (종류, 원래 값)으로 글 속 순서대로 돌려준다.

        예: ("NAME", "Kate Morgan"), ("EMAIL", "kate.morgan@juul.com").

        겹치는 것을 정리하는 방식이 `mask`와 같도록, 바꾸는 대신 그대로 두는(keep) 익명화를 돌려
        실제로 가려질 부분만 얻는다.
        """
        if not self._entities:
            return []
        found = self._analyzer.analyze(text, language="en", entities=self._entities)
        kept = self._anonymizer.anonymize(text, found, {"DEFAULT": OperatorConfig("keep")})
        names = {t.entity: name for name, t in TYPES.items()}
        return [(names[i.entity_type], i.text) for i in sorted(kept.items, key=lambda i: i.start)]


def load_masker(config: dict[str, Any]) -> Masker:
    """config의 PII 종류 목록(`pii.types`)에 있는 종류만 가리는 마스커."""
    return Masker(config["pii"]["spacy_model"], config["pii"]["types"])


@cache
def _analyzer(spacy_model: str) -> AnalyzerEngine:
    """spaCy 모델은 불러오는 데 몇 초 걸리므로 모델마다 한 번만 만든다."""
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.add_recognizer(SpacyRecognizer(supported_entities=["PERSON"]))
    registry.add_recognizer(PatternRecognizer("EMAIL", patterns=[Pattern("email", EMAIL, 1.0)]))
    registry.add_recognizer(PatternRecognizer("PHONE", patterns=[Pattern("phone", PHONE, 1.0)]))
    nlp_engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": spacy_model}])
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])
