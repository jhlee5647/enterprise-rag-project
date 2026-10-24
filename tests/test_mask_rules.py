"""`collect` → `index`: 마스커가 가운데 머리글자 뒤 성, 저자 표기 빈틈, 따옴표 별명을 가리고,
문서 구성 표기·날짜·되풀이 잡음은 가리지 않는다.

테스트 이름의 `tNN`은 티켓 #38 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
사례마다 그 문장만 본문에 든 fixture 문서(`mskb00NN`, NN은 사례 번호)를 색인해 청크를 읽는다.
NER은 실제 spaCy 모델을 쓴다.
"""

from enterprise_rag import mask


def indexed(make_pipeline, doc_id):
    """문서 하나만 `collect` → `index`해 색인된 청크 내용을 돌려준다."""
    pipeline = make_pipeline({("Product Design & Quality", "document"): [doc_id]})
    return pipeline.chunks(doc_id)


def test_t01_surname_after_middle_initial_is_masked(make_pipeline):
    [text] = indexed(make_pipeline, "mskb0001")

    assert "Hoang" not in text
    assert "Joseph" not in text


def test_extra_surname_after_initial_is_masked_when_ner_misses_the_name(make_pipeline):
    # 예: 메일 서명. spaCy가 이름을 못 찾고 저자 표기 정규식이 `Linda C`까지만 잡던 경우
    [text] = indexed(make_pipeline, "mskb0010")

    assert "Alano" not in text
    assert text.endswith("Marketing @ JUUL")


def test_t02_sentence_after_a_word_and_period_is_still_cut(make_pipeline):
    assert indexed(make_pipeline, "mskb0002") == ["Please call [NAME]. Please confirm by Friday."]


def test_t03_author_citation_before_et_al_is_masked(make_pipeline):
    [text] = indexed(make_pipeline, "mskb0003")

    assert "Hajek" not in text


def test_t04_author_citation_at_line_end_is_masked(make_pipeline):
    [text] = indexed(make_pipeline, "mskb0004")

    assert "McKeganey" not in text
    assert text.endswith("Survey of adult smokers")


def test_t05_document_labels_are_not_author_citations(make_pipeline):
    assert indexed(make_pipeline, "mskb0005") == [
        "See Schedule A, Appendix B, and Exhibit C, attached."
    ]


def test_t06_author_citations_are_masked_next_to_a_document_label(make_pipeline):
    assert indexed(make_pipeline, "mskb0006") == ["[NAME], [NAME], and Exhibit A,"]


def test_t07_quoted_nickname_is_masked_with_the_name(make_pipeline):
    [text] = indexed(make_pipeline, "mskb0007")

    assert "Tad" not in text
    assert "Berger" not in text
    assert text.startswith("Chief Deputy Attorney General")


def test_t08_month_year_dates_are_left_as_is(make_pipeline):
    assert indexed(make_pipeline, "mskb0008") == ["Jan-19 Mar-19 Jun-19"]


def test_t09_repeated_noise_is_left_as_is(make_pipeline):
    assert indexed(make_pipeline, "mskb0009") == ["MI MI\nMI MI\nMI MI MI"]


def test_t10_masking_saved_by_masker_version_1_is_not_reused(pipeline, caplog, monkeypatch):
    index_with_masker_version_1(pipeline, monkeypatch)
    caplog.set_level("INFO")
    caplog.clear()

    pipeline.index()

    messages = [r.getMessage() for r in caplog.records]
    assert any("마스킹했다" in m for m in messages)
    assert not any("저장된 마스킹 결과를 썼다" in m for m in messages)


def index_with_masker_version_1(pipeline, monkeypatch):
    """#19 마스커(버전 1)로 색인해 마스킹 결과 파일을 남긴다. 끝나면 지금 버전으로 되돌린다."""
    with monkeypatch.context() as m:
        m.setattr(mask, "VERSION", 1)
        pipeline.index()
