"""`eval`: PII 누락률·오탐률을 정답 자리와 가린 자리(글자 위치)로 센다.

테스트 이름의 `tNN`은 티켓 #37 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
마스커는 가린 자리를 테스트가 정해 주는 가짜로 바꾼다. 지표 정의만 검증하고, 마스커 규칙이
바뀌어도(#38) 이 테스트는 흔들리지 않게 하기 위해서다.
"""

import pytest

from enterprise_rag import mask


class FakeMasker:
    """원문마다 가릴 글자열 목록을 받아, 앞에서부터 아직 가리지 않은 첫 자리를 가린다."""

    def __init__(self, masked_by_text):
        self.masked_by_text = masked_by_text

    def spans(self, text):
        found, start_at = [], {}
        for value in self.masked_by_text.get(text, []):
            start = text.index(value, start_at.get(value, 0))
            start_at[value] = start + len(value)
            found.append(("NAME", start, start + len(value)))
        return sorted(found, key=lambda s: s[1])


@pytest.fixture
def masked(monkeypatch):
    """`masked({원문: [가릴 글자열, ...]})`로 가짜 마스커를 쓰게 한다."""

    def use(masked_by_text):
        monkeypatch.setattr(mask, "load_masker", lambda config: FakeMasker(masked_by_text))

    return use


def chunk(text, pii, doc_id="P1"):
    return {"doc_id": doc_id, "chunk": "0", "text": text, "pii": pii, "split": "test"}


def test_t01_name_partly_masked_counts_as_missed(pipeline, masked):
    text = "Barbara D. Underwood, Attorney General"
    masked({text: ["Barbara D"]})

    run = pipeline.eval([], "t01", pii_answers=[chunk(text, "NAME: Barbara D. Underwood")])

    assert run.metrics["pii_miss_rate"] == 1.0
    assert run.metrics["pii_false_positive_rate"] == 0.0


def test_t02_name_masked_with_next_word_is_not_false_positive(pipeline, masked):
    text = "Durbin said the bill"
    masked({text: ["Durbin said"]})

    run = pipeline.eval([], "t02", pii_answers=[chunk(text, "NAME: Durbin")])

    assert run.metrics["pii_miss_rate"] == 0.0
    assert run.metrics["pii_false_positive_rate"] == 0.0


def test_t03_only_the_unmasked_repeat_counts_as_missed(pipeline, masked):
    text = "Tiffany called. Later Tiffany wrote back."
    masked({text: ["Tiffany"]})  # 첫 번째만 가림

    run = pipeline.eval([], "t03", pii_answers=[chunk(text, "NAME: Tiffany; NAME: Tiffany")])

    assert run.metrics["pii_miss_rate"] == 0.5


def test_t04_masked_place_outside_every_answer_is_false_positive(pipeline, masked):
    text = "Kate Morgan presented the Bebop launch."
    masked({text: ["Kate Morgan", "Bebop"]})

    run = pipeline.eval([], "t04", pii_answers=[chunk(text, "NAME: Kate Morgan")])

    assert run.metrics["pii_false_positive_rate"] == 0.5


def test_t05_value_inside_a_longer_word_is_not_an_answer_place(pipeline, masked):
    text = "Ed met Eddy"
    masked({text: []})

    run = pipeline.eval([], "t05", pii_answers=[chunk(text, "NAME: Ed")])

    # 정답 자리는 앞의 Ed 하나: 1 ÷ 1
    assert run.metrics["pii_miss_rate"] == 1.0


def test_t06_short_value_inside_a_longer_answer_is_not_counted_again(pipeline, masked):
    text = "Denise Hannigan called. Hannigan left."
    masked({text: ["Denise Hannigan"]})

    run = pipeline.eval(
        [], "t06", pii_answers=[chunk(text, "NAME: Denise Hannigan; NAME: Hannigan")]
    )

    # 정답 자리는 Denise Hannigan, 뒤의 Hannigan 두 개. 뒤의 것이 남음: 1 ÷ 2
    assert run.metrics["pii_miss_rate"] == 0.5


def test_t07_answer_value_missing_from_text_stops_eval(pipeline, masked, capsys):
    # 예: 사람이 정답을 고치다가 이름 철자를 틀리게 적은 경우
    masked({})
    rows = [
        chunk("Please call Kate Morgan.", "NAME: Kate Morgan"),
        chunk("John Smith signed the letter.", "NAME: Jon Smith", doc_id="P2"),
    ]

    code = pipeline.eval([], "t07", pii_answers=rows, expect_code=2)

    assert code == 2
    err = capsys.readouterr().err
    assert "P2" in err
    assert "Jon Smith" in err


def test_t08_no_answers_and_nothing_masked_leaves_both_metrics_empty(pipeline, masked):
    text = "The FDA reviewed the application."
    masked({text: []})

    run = pipeline.eval([], "t08", pii_answers=[chunk(text, "")])

    assert run.metrics["pii_miss_rate"] is None
    assert run.metrics["pii_false_positive_rate"] is None
