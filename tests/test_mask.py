"""`collect` → `index` → `ask`: 색인 전 PII 마스킹을 실제 spaCy 모델로 확인한다.

테스트 이름의 `tNN`은 티켓 #4 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
문서 P는 pnot0001이다.
"""

PII = ["Kate Morgan", "kate.morgan@juul.com", "555-0134"]
QUESTION = "influencer budget"


def test_t01_document_body_is_masked_before_indexing(pipeline):
    asked = pipeline("marketing", QUESTION)

    assert "pnot0001" in asked.retrieved_ids
    [chunk] = pipeline.chunks("pnot0001")
    for label in ("[NAME]", "[EMAIL]", "[PHONE]"):
        assert label in chunk
    for value in PII:
        assert value not in chunk


def test_t02_answer_citations_and_prompt_have_no_pii(pipeline):
    asked = pipeline("marketing", QUESTION)

    assert "pnot0001" in asked.cited_ids
    [prompt] = pipeline.prompts
    assert "[NAME]" in prompt
    for value in PII:
        assert value not in asked.answer
        assert value not in str(asked.citations)
        assert value not in prompt


def test_t03_citation_title_is_masked(pipeline):
    asked = pipeline("marketing", QUESTION)

    [citation] = [c for c in asked.citations if c["doc_id"] == "pnot0001"]
    assert citation["title"] == "Budget note from [NAME]"


def test_t04_email_on_other_company_domain_is_masked(pipeline):
    assert pipeline.chunks("jdoe0001") == ["Send it to [EMAIL]."]


def test_t05_text_without_pii_is_unchanged(pipeline):
    assert pipeline.chunks("fdar0001") == ["The FDA requested additional data."]


def test_t06_pii_types_from_config_leave_names_unmasked(pipeline):
    pipeline.index(pii_types=["EMAIL", "PHONE"])
    asked = pipeline("marketing", QUESTION)

    assert "pnot0001" in asked.retrieved_ids
    [chunk] = pipeline.chunks("pnot0001")
    assert "Kate Morgan" in chunk
    assert "[EMAIL]" in chunk
    assert "[PHONE]" in chunk
    assert "[NAME]" not in chunk
