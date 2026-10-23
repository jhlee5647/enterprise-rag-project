"""PII 마스커: 실명(NER), 이메일·전화번호(정규식)를 찾아 가명 토큰으로 바꾼다. 색인 전에 쓴다."""

import re
from functools import cache
from typing import Any, NamedTuple

import spacy
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NerModelConfiguration, NlpArtifacts, SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# 마스커 버전: 마스킹 결과가 바뀔 수 있는 변경을 할 때 올린다. 바뀌면 `index`가 저장된 마스킹
# 결과를 쓰지 않는다. 예: 정규식(EMAIL, PHONE, AUTHOR), 이름 후보 규칙, 불러올 spaCy 단계,
# spaCy·Presidio 버전 올리기
VERSION = 1

# spaCy에서 불러올 단계. en_core_web_lg의 `ner`은 자기 안에 tok2vec을 따로 갖고 있어 공용
# `tok2vec` 단계 없이도 결과가 같다(공용 `tok2vec`은 `tagger`, `parser`만 쓴다)
SPACY_PIPES = ["ner"]
# spaCy가 찾는 PERSON 외 개체. 가리지 않으므로 Presidio가 개체마다 찍는 경고를 끈다
IGNORED_LABELS = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
    "NORP", "ORDINAL", "ORG", "PERCENT", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART",
]  # fmt: skip

# `@` 자리에 OCR로 깨진 `©`, `(c)`, `(c~`도 받는다: support©juulvapor.com
EMAIL = (
    r"(?<![\w.%+-])[A-Za-z0-9._%+-]+(?:@|©|\(c\)|\(c~)"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)
# 북미 번호: 415-555-0134, (415) 555-0134, 415.555.0134, +1 415 555 0134
PHONE = r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
# 논문 저자 표기: 성(하이픈 허용) + 머리글자 1~2개 + `,` `.` `;`. 예: Severson HH, Hartmann-Boyce J,
# 머리글자가 로마 숫자(I, V, X)뿐이면 임상 단계 표기(Phase II,)로 보고 빼놓는다.
AUTHOR = r"(?<![\w-])[A-Z][a-z]+(?:-[A-Z][a-z]+)* (?![IVX]{1,2}[,.;])[A-Z]{1,2}(?=[,.;])"

# spaCy가 찾은 이름 후보를 나누는 자리(줄바꿈, ` and `)와, 이름 뒤에 붙은 문장의 시작(`. ` + 대문자)
NAME_SEPARATOR = re.compile(r"\s*\n\s*| and ")
SENTENCE_AFTER_NAME = re.compile(r"\.\s[A-Z]")
# 이 중 하나라도 맞는 이름 후보는 OCR 잡음으로 보고 버린다
NOT_NAME = [
    # 라틴 문자(확장 포함), 숫자, 공백, `.` `-` `'` 외의 글자
    re.compile(r"[^A-Za-zÀ-ÖØ-öø-ɏ0-9\s.'-]"),
    re.compile(r"^(?!.*[^\W\d_]{2})", re.DOTALL),  # 알파벳 2글자 이상인 단어가 없음
    re.compile(r"([^\W\d_]{2,3})\1{2,}"),  # 같은 2~3글자가 3번 이상 연달아 반복됨
    re.compile(r"^(?!.*(?<!\w)[A-Z])", re.DOTALL),  # 대문자로 시작하는 단어가 없음
]


class PiiType(NamedTuple):
    entity: str  # Presidio 개체 이름
    token: str  # 가명 토큰


# config의 PII 종류 목록에 쓰는 이름 → PII 종류
TYPES = {
    "NAME": PiiType("PERSON", "[NAME]"),
    "EMAIL": PiiType("EMAIL", "[EMAIL]"),
    "PHONE": PiiType("PHONE", "[PHONE]"),
}
# Presidio 개체 이름 → config의 PII 종류 이름
NAMES = {t.entity: name for name, t in TYPES.items()}


class Masker:
    def __init__(self, spacy_model: str, types: list[str]):
        check_types(types)
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
        return [(NAMES[i.entity_type], i.text) for i in sorted(kept.items, key=lambda i: i.start)]


def check_types(types: list[str]) -> None:
    """PII 종류 목록에 모르는 종류가 있으면 ValueError."""
    unknown = [t for t in types if t not in TYPES]
    if unknown:
        raise ValueError(f"알 수 없는 PII 종류: {unknown} (쓸 수 있는 종류: {list(TYPES)})")


def load_masker(config: dict[str, Any]) -> Masker:
    """config의 PII 종류 목록(`pii.types`)에 있는 종류만 가리는 마스커."""
    return Masker(config["pii"]["spacy_model"], config["pii"]["types"])


class NameRecognizer(SpacyRecognizer):
    """spaCy가 PERSON으로 찾은 이름 후보의 경계를 고치고, OCR 잡음인 후보를 버린다."""

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
        return [
            piece
            for found in super().analyze(text, entities, nlp_artifacts)
            for piece in _name_pieces(text, found)
        ]


def _name_pieces(text: str, found: RecognizerResult) -> list[RecognizerResult]:
    """이름 후보를 줄바꿈과 ` and `에서 나누고, 조각마다 `. ` 뒤 문장을 잘라 낸 뒤 잡음이 아닌
    조각만 돌려준다. 예: `Robert\\nKrulik` → `Robert`, `Krulik`
    """
    candidate = text[found.start : found.end]
    spans, start = [], 0
    for separator in NAME_SEPARATOR.finditer(candidate):
        spans.append((start, separator.start()))
        start = separator.end()
    spans.append((start, len(candidate)))

    pieces = []
    for start, end in spans:
        if sentence := SENTENCE_AFTER_NAME.search(candidate, start, end):
            end = sentence.start()
        name = candidate[start:end].strip()
        if not name or any(rule.search(name) for rule in NOT_NAME):
            continue
        begin = found.start + candidate.index(name, start)
        pieces.append(
            RecognizerResult(
                found.entity_type,
                begin,
                begin + len(name),
                found.score,
                found.analysis_explanation,
                found.recognition_metadata,
            )
        )
    return pieces


class NerOnlyNlpEngine(SpacyNlpEngine):
    """spaCy 모델에서 `SPACY_PIPES` 단계만 불러온다. 나머지 단계는 메모리에도 올리지 않는다.
    Presidio의 원래 `load`가 하는 GPU 켜기와 모델 자동 내려받기는 하지 않는다.
    """

    def load(self) -> None:
        self.nlp = {}
        for m in self.models:
            name = m["model_name"]
            unused = [c for c in spacy.info(name)["components"] if c not in SPACY_PIPES]
            self.nlp[m["lang_code"]] = spacy.load(name, exclude=unused)


@cache
def _analyzer(spacy_model: str) -> AnalyzerEngine:
    """spaCy 모델은 불러오는 데 몇 초 걸리므로 모델마다 한 번만 만든다."""
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.add_recognizer(NameRecognizer(supported_entities=["PERSON"]))
    # 저자 표기는 대소문자로 판정하므로 Presidio 기본 플래그(IGNORECASE 등)를 모두 끈다
    author = Pattern("author", AUTHOR, 1.0)
    registry.add_recognizer(
        PatternRecognizer(
            "PERSON", name="AuthorRecognizer", patterns=[author], global_regex_flags=0
        )
    )
    registry.add_recognizer(PatternRecognizer("EMAIL", patterns=[Pattern("email", EMAIL, 1.0)]))
    registry.add_recognizer(PatternRecognizer("PHONE", patterns=[Pattern("phone", PHONE, 1.0)]))
    nlp_engine = NerOnlyNlpEngine(
        models=[{"lang_code": "en", "model_name": spacy_model}],
        ner_model_configuration=NerModelConfiguration(labels_to_ignore=IGNORED_LABELS),
    )
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])
