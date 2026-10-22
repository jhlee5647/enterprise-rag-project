"""PII 마스커: 실명(NER), 이메일·전화번호(정규식)를 찾아 고정 라벨로 바꾼다. 색인 전에 쓴다."""

from functools import cache
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

EMAIL = r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
# 북미 번호: 415-555-0134, (415) 555-0134, 415.555.0134, +1 415 555 0134
PHONE = r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"

# PII 종류 → (Presidio 개체 이름, 고정 라벨)
TYPES = {
    "NAME": ("PERSON", "[NAME]"),
    "EMAIL": ("EMAIL", "[EMAIL]"),
    "PHONE": ("PHONE", "[PHONE]"),
}


class Masker:
    def __init__(self, spacy_model: str, types: list[str]):
        unknown = [t for t in types if t not in TYPES]
        if unknown:
            raise ValueError(f"알 수 없는 PII 종류: {unknown} (쓸 수 있는 종류: {list(TYPES)})")
        self._analyzer = _analyzer(spacy_model)
        self._anonymizer = AnonymizerEngine()
        # PII 종류 목록에 있는 것만 찾는다. 목록이 비면 아무것도 바꾸지 않는다.
        self._entities = [TYPES[t][0] for t in types]
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": label})
            for entity, label in TYPES.values()
        }

    def mask(self, text: str) -> str:
        if not self._entities:
            return text
        found = self._analyzer.analyze(text, language="en", entities=self._entities)
        return self._anonymizer.anonymize(text, found, self._operators).text


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
