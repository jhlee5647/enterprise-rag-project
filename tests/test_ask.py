"""`collect` → `index` → `ask`: 부서별 질의응답을 가짜 임베딩과 가짜 LLM으로 확인한다.

테스트 이름의 `tNN`은 티켓 #3 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
문서 A는 inca0001, B는 pmtb0001, C는 agec0001이다.
"""

NOT_FOUND = "No relevant documents were found."


def test_t01_answer_cites_document_id_bates_and_idl_link(pipeline):
    asked = pipeline("marketing", "influencer campaign budget")

    assert {
        "doc_id": "inca0001",
        "bates": "JLI00000501",
        "url": "https://www.industrydocuments.ucsf.edu/docs/inca0001",
    } in asked.citations


def test_t02_acl_filter_excludes_other_department_documents_before_search(pipeline):
    asked = pipeline("marketing", "PMTA submission timeline")

    assert "pmtb0001" not in asked.retrieved_ids
    assert asked.retrieved_ids  # 다른 청크는 검색됐지만 근거가 되지 못했다
    assert asked.answer == NOT_FOUND
    assert asked.citations == []


def test_t03_department_with_access_gets_answer(pipeline):
    asked = pipeline("regulatory", "PMTA submission timeline")

    assert "pmtb0001" in asked.cited_ids


def test_t04_document_allowed_to_two_departments_is_answered_for_both(pipeline):
    marketing = pipeline("marketing", "age verification policy")
    regulatory = pipeline("regulatory", "age verification policy")

    assert "agec0001" in marketing.cited_ids
    assert "agec0001" in regulatory.cited_ids


def test_t05_department_not_allowed_gets_not_found(pipeline):
    asked = pipeline("product", "age verification policy")

    assert asked.answer == NOT_FOUND
    assert asked.citations == []


def test_t06_no_evidence_and_no_access_give_identical_response(pipeline):
    no_evidence = pipeline("marketing", "quantum chromodynamics")
    no_access = pipeline("marketing", "PMTA submission timeline")

    assert no_evidence.answer == no_access.answer == NOT_FOUND


def test_t07_acl_filter_off_retrieves_other_department_documents(pipeline):
    asked = pipeline("marketing", "PMTA submission timeline", acl_filter=False)

    assert "pmtb0001" in asked.retrieved_ids


def test_t08_unknown_department_fails_with_valid_departments(pipeline):
    asked = pipeline("sales", "budget")

    assert asked.code != 0
    for department in ("marketing", "regulatory", "product"):
        assert department in asked.err
