"""`collect` → `index`: 마스커가 OCR 잡음과 잘린 이름을 덜 가리고, 저자 표기와 깨진 이메일을
더 가린다.

테스트 이름의 `tNN`은 티켓 #19 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
사례마다 그 문장만 본문에 든 fixture 문서(`mska00NN`, NN은 사례 번호)를 색인해 청크를 읽는다.
문서 P는 pnot0001이다. NER은 실제 spaCy 모델을 쓴다.
"""

import pytest


def indexed(make_pipeline, doc_id):
    """문서 하나만 `collect` → `index`해 색인된 청크 내용을 돌려준다."""
    pipeline = make_pipeline({("Product Design & Quality", "document"): [doc_id]})
    return pipeline.chunks(doc_id)


@pytest.mark.parametrize(
    "doc_id, text",
    [
        # t01: 라틴 문자 외 글자
        ("mska0001", "工愚二－口印昂、－－三…n－「「一『一。－山s 二＝山＝「。「＃n鲎"),
        # t02: 알파벳 2글자 이상 단어 없음
        ("mska0002", ".,r"),
        # t03: 같은 글자 반복
        ("mska0003", "____ Humumumumu tlIPPI IPPIRIPPIP"),
        # t04: 대문자로 시작하는 단어 없음
        ("mska0004", "ana aelete"),
    ],
    ids=["t01", "t02", "t03", "t04"],
)
def test_t01_t04_ocr_noise_is_left_as_is(make_pipeline, doc_id, text):
    assert indexed(make_pipeline, doc_id) == [text]


def test_t05_name_candidate_is_split_at_line_break(make_pipeline):
    assert indexed(make_pipeline, "mska0005") == ["[NAME]\n[NAME] reviewed the BOM."]


def test_t06_name_candidate_is_split_at_and(make_pipeline):
    assert indexed(make_pipeline, "mska0006") == ["Hi [NAME] and [NAME], the call is at noon."]


def test_t07_sentence_after_period_is_cut_from_name(make_pipeline):
    assert indexed(make_pipeline, "mska0007") == ["[NAME]. Antidepressants for smoking cessation."]


def test_t08_author_citations_are_masked(make_pipeline):
    assert indexed(make_pipeline, "mska0008") == ["[NAME], [NAME], [NAME]."]


def test_t09_roman_numeral_is_not_an_author_citation(make_pipeline):
    assert indexed(make_pipeline, "mska0009") == ["Phase II, results were positive."]


def test_t10_email_with_copyright_sign_for_at_is_masked(make_pipeline):
    assert indexed(make_pipeline, "mska0010") == ["Write to [EMAIL] today."]


def test_t11_email_with_c_tilde_for_at_is_masked(make_pipeline):
    assert indexed(make_pipeline, "mska0011") == ["Unsubscribe: [EMAIL]"]


def test_t12_document_p_is_still_masked(make_pipeline):
    assert indexed(make_pipeline, "pnot0001") == [
        "Please contact [NAME] at [EMAIL] or [PHONE] about the influencer budget."
    ]
