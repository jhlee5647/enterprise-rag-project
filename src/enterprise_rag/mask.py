"""PII 마스커: 실명(NER), 이메일·전화번호(정규식)를 찾아 가명 토큰으로 바꾼다. 색인 전에 쓴다."""

import re
from functools import cache
from typing import Any, NamedTuple

import spacy
import torch
from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NerModelConfiguration, NlpArtifacts, SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from spacy.lang.en.stop_words import STOP_WORDS

# 마스커 버전: 마스킹 결과가 바뀔 수 있는 변경을 할 때 올린다. 바뀌면 `index`가 저장된 마스킹
# 결과를 쓰지 않는다. 예: 정규식(EMAIL, PHONE, AUTHOR), 이름 후보 규칙, 불러올 spaCy 단계,
# spaCy·Presidio 버전 올리기
VERSION = 2

# spaCy에서 불러올 단계. `ner`은 개체가 문장 경계를 넘지 않게 하므로, 문장 경계를 정하는
# `parser`(와 그 입력인 `tok2vec`)가 없으면 결과가 달라진다. 나머지 단계는 결과와 상관없다
SPACY_PIPES = ["tok2vec", "parser", "ner"]
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
# 따옴표 별명이 낀 이름: 이름 + "별명" + 성. 예: Edmund "Tad" Berger. 곧은·둥근 따옴표 모두 받는다
NICKNAME = r'(?<![\w-])[A-Z][a-z]+ ["“][A-Z][a-z]+["”] [A-Z][a-z]+(?:-[A-Z][a-z]+)*\b'
# 이름 찾기 모델: spaCy(기본) 또는 GLiNER (#44)
NAME_MODELS = ["spacy", "gliner"]
GLINER_MODEL = "urchade/gliner_medium-v2.1"
# GLiNER에 함께 주는 레이블. 첫 레이블로 잡힌 것만 이름 후보로 쓴다. "person" 대신 "person name"으로
# 이름 자체를 묻고, 직함과 회사를 함께 물어 역할 명사(Sales Rep, Distributor)가 그쪽으로 가게 한다.
# 개발용 50청크에서 정했다: NAME 누락률 0.243, 오탐률 0.196 ("person"만이면 0.243 / 0.316)
GLINER_LABELS = ["person name", "job title", "organization"]
GLINER_THRESHOLD = 0.5
# GLiNER에 한 번에 넘기는 글 조각의 최대 길이(글자). 줄 단위로 끊는다
GLINER_WINDOW = 1200
# 사람으로 잡혔어도 바로 앞에 한정사가 있으면 실명이 아니라 역할 명사로 본다: the Supplier,
# each User
ROLE_DETERMINER = re.compile(
    r"\b(?:the|a|an|each|every|any|all|no|this|that|these|those|our|your|their|its|such)\s+\Z",
    re.IGNORECASE,
)
# 이름 앞 직함. 예: Sen. Mark Warner, Dr. Smith
TITLE = re.compile(r"(?:Sen|Rep|Gov|Dr|Mr|Mrs|Ms|Prof|Hon|Judge|Rev)\.?\s+")
# 저자 표기처럼 보여도 문서 구성 표기(Exhibit A, Appendix B, Cohort C)의 첫 단어이면 이름이 아니다.
# 9,000건에서 저자 표기 모양에 걸린 첫 단어 중 이름이 될 수 없는 것만 넣는다 (#38). 성으로도
# 쓰이는 단어(Box, Stage, Round, Part, Grade, Batch, Wave)는 넣지 않는다: `Box GE,`는 저자 표기다
DOCUMENT_WORDS = [
    "Annex", "Appendix", "Area", "Attachment", "Ave", "Category", "Circle", "Class", "Cohort",
    "Concept", "Cumulative", "Exhibit", "Figure", "Group", "Item", "Level", "Option", "Package",
    "Pair", "Phase", "Plan", "Retailer", "Rev", "Room", "Schedule", "Section", "Series", "Step",
    "Suite", "Tab", "Table", "Tier", "Type", "Version", "Zone",
]  # fmt: skip
# 논문 저자 표기: 성(하이픈, McKeganey 같은 가운데 대문자 허용) + 머리글자 1~2개 + `,` `.` `;`
# 또는 ` et al`. 예: Severson HH, Hartmann-Boyce J, Hajek P et al.
# 줄 끝에서 끝나면 머리글자 1개일 때만 본다(McKeganey N). 2개이면 Total US, Flex BG 같은 표 칸이
# 많기 때문이다. 머리글자가 로마 숫자(I, V, X)뿐이면 임상 단계 표기(Phase II,)로 보고 빼놓는다.
# 머리글자 1개 + `. ` 뒤에 대문자 단어가 오면 가운데 머리글자로 보고 그 단어(성)까지 잡는다
# (Linda C. Alano). 뒤 단어가 기능어(The, We 등)이면 문장의 시작으로 보고 머리글자까지만 잡는다
_SENTENCE_STARTS = "|".join(sorted(w.title() for w in STOP_WORDS if w.isalpha()))
AUTHOR = (
    rf"(?<![\w-])(?!(?:{'|'.join(DOCUMENT_WORDS)}) )[A-Z][a-z]+(?:[A-Z][a-z]+)?(?:-[A-Z][a-z]+)* "
    r"(?![IVX]{1,2}\b)"
    rf"(?:[A-Z]\. (?!(?:{_SENTENCE_STARTS})\b)[A-Z][a-z]+(?:[A-Z][a-z]+)?(?:-[A-Z][a-z]+)*\b"
    r"|[A-Z]{1,2}(?=[,.;]| et al\b)|[A-Z](?=[ \t]*(?:\r?\n|\Z)))"
)

# spaCy가 찾은 이름 후보를 나누는 자리(줄바꿈, ` and `)와, 이름 뒤에 붙은 문장의 시작(`. ` + 대문자)
NAME_SEPARATOR = re.compile(r"\s*\n\s*| and ")
SENTENCE_AFTER_NAME = re.compile(r"\.\s([A-Z][a-z]*)")
# 마침표 앞이 대문자 한 글자(가운데 머리글자)이면 뒤를 자르지 않는다: `Joseph N. Hoang`의 `N.` 뒤를
# 자르면 성이 드러난다. 다만 뒤 단어가 기능어(`The`, `We` 등)이면 문장의 시작이 분명하므로 자른다
INITIAL_BEFORE_PERIOD = re.compile(r"(?<![^\W\d_])[A-Z]$")
# 표 머리글의 월-연 날짜. 예: Jan-19, Sept-20
MONTH_YEAR = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)-\d{2}"
# 이 중 하나라도 맞는 이름 후보는 OCR 잡음이나 날짜로 보고 버린다
NOT_NAME = [
    # 라틴 문자(확장 포함), 숫자, 공백, `.` `-` `'` 외의 글자
    re.compile(r"[^A-Za-zÀ-ÖØ-öø-ɏ0-9\s.'-]"),
    re.compile(r"^(?!.*[^\W\d_]{2})", re.DOTALL),  # 알파벳 2글자 이상인 단어가 없음
    re.compile(r"([^\W\d_]{2,3})\1{2,}"),  # 같은 2~3글자가 3번 이상 연달아 반복됨
    re.compile(r"^(?!.*(?<!\w)[A-Z])", re.DOTALL),  # 대문자로 시작하는 단어가 없음
    re.compile(r"^([^\W\d_]{2,3})(?:\s+\1)+$", re.IGNORECASE),  # 같은 2~3글자 단어만 되풀이됨
    re.compile(rf"^{MONTH_YEAR}(?:\s+{MONTH_YEAR})*$"),  # 날짜(Jan-19)만 있음
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
    def __init__(self, spacy_model: str, types: list[str], name_model: str = "spacy"):
        check_types(types)
        check_name_model(name_model)
        self._analyzer = _analyzer(spacy_model, name_model)
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
        """
        return [(kind, text[start:end]) for kind, start, end in self.masked_places(text)]

    def masked_places(self, text: str) -> list[tuple[str, int, int]]:
        """`mask`가 가리는 자리(가린 자리)를 (종류, 시작, 끝)으로 글 속 순서대로 돌려준다.
        끝은 포함하지 않는다. 예: "Call Kate Morgan" → [("NAME", 5, 16)].

        겹치는 것을 정리하는 방식이 `mask`와 같도록, 바꾸는 대신 그대로 두는(keep) 익명화를 돌려
        실제로 가려질 부분만 얻는다.
        """
        if not self._entities:
            return []
        found = self._analyzer.analyze(text, language="en", entities=self._entities)
        kept = self._anonymizer.anonymize(text, found, {"DEFAULT": OperatorConfig("keep")})
        items = sorted(kept.items, key=lambda i: i.start)
        return [(NAMES[i.entity_type], i.start, i.end) for i in items]


def check_types(types: list[str]) -> None:
    """PII 종류 목록에 모르는 종류가 있으면 ValueError."""
    unknown = [t for t in types if t not in TYPES]
    if unknown:
        raise ValueError(f"알 수 없는 PII 종류: {unknown} (쓸 수 있는 종류: {list(TYPES)})")


def load_masker(config: dict[str, Any]) -> Masker:
    """config의 PII 종류 목록(`pii.types`)에 있는 종류만, 이름 찾기 모델(`pii.name_model`)로 가리는
    마스커."""
    pii = config["pii"]
    return Masker(pii["spacy_model"], pii["types"], name_model(config))


def name_model(config: dict[str, Any]) -> str:
    """config의 이름 찾기 모델. 지정하지 않으면 spaCy다."""
    return config["pii"].get("name_model", "spacy")


def check_name_model(name: str) -> None:
    """모르는 이름 찾기 모델이면 ValueError."""
    if name not in NAME_MODELS:
        raise ValueError(f"알 수 없는 이름 찾기 모델: {name} (쓸 수 있는 값: {NAME_MODELS})")


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


class GlinerNameRecognizer(EntityRecognizer):
    """GLiNER가 사람으로 찾은 자리에서 직함을 떼고 마스커의 보정 규칙(`_name_pieces`)을 적용."""

    def __init__(self) -> None:
        super().__init__(supported_entities=["PERSON"], name="GlinerNameRecognizer")

    def load(self) -> None:
        pass

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
        model = _gliner()
        results = []
        for offset, piece in _windows(text, GLINER_WINDOW):
            found = model.predict_entities(piece, GLINER_LABELS, threshold=GLINER_THRESHOLD)
            for e in found:
                if e["label"] != GLINER_LABELS[0]:
                    continue
                start, end = offset + e["start"], offset + e["end"]
                # GLiNER는 직함까지 이름으로 잡는다(`Sen. Mark Warner`). 떼지 않으면 보정 규칙이
                # `Sen.` 뒤를 다음 문장으로 보고 잘라 이름을 버린다
                if title := TITLE.match(text, start, end):
                    start = title.end()
                if ROLE_DETERMINER.search(text[max(0, start - 12) : start]):
                    continue
                found_name = RecognizerResult("PERSON", start, end, e["score"])
                results += _name_pieces(text, found_name)
        return results


@cache
def _gliner() -> Any:
    """GLiNER 모델은 불러오는 데 몇 초 걸리므로 프로세스마다 한 번만 불러 둔다. GPU가 있으면 씀."""
    from gliner import GLiNER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GLiNER.from_pretrained(GLINER_MODEL).to(device)


def _windows(text: str, size: int) -> list[tuple[int, str]]:
    """글을 줄 단위로 `size` 글자 이하의 조각으로 나눠 (시작 위치, 조각)으로 돌려준다.

    예: size 10이면 "Kate\nMorgan said\n" → [(0, "Kate\n"), (5, "Morgan said\n")].
    """
    out, start, end = [], 0, 0
    for line in re.finditer(r"[^\n]*\n|[^\n]+$", text):
        if line.end() - start > size and end > start:
            out.append((start, text[start:end]))
            start = end
        end = line.end()
        while end - start > size:  # 한 줄이 너무 길면 자른다
            out.append((start, text[start : start + size]))
            start += size
    if end > start:
        out.append((start, text[start:end]))
    return out


def _sentence_after_name(candidate: str, start: int, end: int) -> int | None:
    """이름 후보 조각 [start, end)에서 이름 뒤에 붙은 문장이 시작하는 `.`의 위치. 없으면 None.

    예: `Kevin Burns. Please` → `.`의 위치, `Joseph N. Hoang` → None, `Kevin B. The` → `.`의 위치.
    """
    for m in SENTENCE_AFTER_NAME.finditer(candidate, start, end):
        initial = INITIAL_BEFORE_PERIOD.search(candidate, start, m.start())
        if initial and m.group(1).lower() not in STOP_WORDS:
            continue
        return m.start()
    return None


def _name_pieces(text: str, found: RecognizerResult) -> list[RecognizerResult]:
    """이름 후보를 줄바꿈과 ` and `에서 나누고, 조각마다 이름 뒤에 붙은 문장을 잘라 낸 뒤
    잡음이 아닌 조각만 돌려준다. 자르는 자리는 `_sentence_after_name`이 정한다.
    예: `Robert\\nKrulik` → `Robert`, `Krulik`
    """
    candidate = text[found.start : found.end]
    spans, start = [], 0
    for separator in NAME_SEPARATOR.finditer(candidate):
        spans.append((start, separator.start()))
        start = separator.end()
    spans.append((start, len(candidate)))

    pieces = []
    for start, end in spans:
        if (sentence := _sentence_after_name(candidate, start, end)) is not None:
            end = sentence
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
    """spaCy 모델에서 `pipes` 단계(기본 `SPACY_PIPES`)만 불러온다. 나머지 단계는 메모리에도 올리지
    않는다. Presidio의 원래 `load`가 하는 GPU 켜기와 모델 자동 내려받기는 하지 않는다.
    """

    def __init__(self, pipes: list[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.pipes = SPACY_PIPES if pipes is None else pipes

    def load(self) -> None:
        self.nlp = {}
        for m in self.models:
            name = m["model_name"]
            unused = [c for c in spacy.info(name)["components"] if c not in self.pipes]
            self.nlp[m["lang_code"]] = spacy.load(name, exclude=unused)


@cache
def _analyzer(spacy_model: str, name_model: str) -> AnalyzerEngine:
    """spaCy·GLiNER 모델은 불러오는 데 몇 초 걸리므로 모델마다 한 번만 만든다."""
    registry = RecognizerRegistry(supported_languages=["en"])
    if name_model == "gliner":
        registry.add_recognizer(GlinerNameRecognizer())
    else:
        registry.add_recognizer(NameRecognizer(supported_entities=["PERSON"]))
    # 저자 표기와 별명은 대소문자로 판정하므로 Presidio 기본 플래그(IGNORECASE 등)를 모두 끈다
    name_patterns = [Pattern("author", AUTHOR, 1.0), Pattern("nickname", NICKNAME, 1.0)]
    registry.add_recognizer(
        PatternRecognizer(
            "PERSON", name="NamePatternRecognizer", patterns=name_patterns, global_regex_flags=0
        )
    )
    registry.add_recognizer(PatternRecognizer("EMAIL", patterns=[Pattern("email", EMAIL, 1.0)]))
    registry.add_recognizer(PatternRecognizer("PHONE", patterns=[Pattern("phone", PHONE, 1.0)]))
    nlp_engine = NerOnlyNlpEngine(
        # GLiNER가 이름을 찾으면 spaCy는 Presidio가 쓰는 토큰 나누기만 한다
        pipes=[] if name_model == "gliner" else None,
        models=[{"lang_code": "en", "model_name": spacy_model}],
        ner_model_configuration=NerModelConfiguration(labels_to_ignore=IGNORED_LABELS),
    )
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])
