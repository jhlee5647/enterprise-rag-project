"""`collect` 명령: 가짜 IDL 서버를 상대로 실행하고 문서 표와 문서 ID 목록을 확인한다.

테스트 이름의 `tNN`은 티켓 #2 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
"""

MARKETING_EMAIL = ("Marketing", "email")
MARKETING_DOCUMENT = ("Marketing", "document")
REGULATORY_EMAIL = ("Government & Public Affairs", "email")
REGULATORY_DOCUMENT = ("Government & Public Affairs", "document")


def test_t01_marketing_email_is_allowed_to_marketing(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0001"]})

    assert collected.by_id["mkea0001"]["departments"] == ["marketing"]


def test_t02_two_mapped_topics_are_allowed_to_both_departments(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["shrm0001"]})

    assert collected.by_id["shrm0001"]["departments"] == ["marketing", "regulatory"]


def test_t03_unmapped_topic_value_is_ignored(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0002"]})

    assert collected.by_id["mkea0002"]["departments"] == ["marketing"]


def test_t04_t05_documents_without_allowed_departments_are_excluded(run_collect):
    # flav0001: 토픽이 `Flavors`뿐, notp0001: `topic` 필드 없음. 할당량 2를 차지하지 않는다.
    collected = run_collect({MARKETING_EMAIL: ["flav0001", "notp0001", "mkea0001", "mkea0003"]})

    assert collected.sample_ids == ["mkea0001", "mkea0003"]
    assert set(collected.by_id) == {"mkea0001", "mkea0003"}


def test_t06_topic_matching_ignores_case_and_trailing_separator(run_collect):
    collected = run_collect({("Product Design & Quality", "presentation"): ["prdp0001"]})

    assert collected.by_id["prdp0001"]["departments"] == ["product"]


def test_t07_body_falls_back_to_txt_when_ocr_is_missing(run_collect):
    collected = run_collect({REGULATORY_DOCUMENT: ["txtd0001"]})

    assert collected.by_id["txtd0001"]["text"] == (
        "PMTA submission outline\n1. Product description\n2. Toxicology\nJLI00000203\n"
    )


def test_t08_document_without_body_is_skipped_and_logged(run_collect, caplog):
    # nobd0001: S3에 .ocr도 .txt도 없다. 할당량을 차지하지 않고 수집은 계속된다.
    collected = run_collect({MARKETING_EMAIL: ["nobd0001", "mkea0001", "mkea0003"]})

    assert collected.sample_ids == ["mkea0001", "mkea0003"]
    assert any("nobd0001" in r.getMessage() and "본문" in r.getMessage() for r in caplog.records)


def test_extra_document_with_too_long_body_is_skipped_and_logged(run_collect):
    # 본문 길이: mkea0002 110자, mkea0001 111자, mkea0003 90자. 한도와 같은 90자는 수집한다.
    # 건너뛴 문서는 할당량을 차지하지 않아 다음 후보로 채운다.
    collected = run_collect(
        {MARKETING_EMAIL: ["mkea0002", "mkea0001", "mkea0003"]}, quota=1, max_chars=90
    )

    assert collected.sample_ids == ["mkea0003"]
    assert any(
        "mkea0002: 본문이 110자로 한도 90자를 넘어 건너뛴다" in line for line in collected.log_lines
    )


def test_t09_document_picked_in_two_cells_is_stored_once(run_collect):
    collected = run_collect(
        {
            MARKETING_EMAIL: ["shrm0001", "mkea0001"],
            REGULATORY_EMAIL: ["shrm0001", "rega0001"],
        }
    )

    assert collected.sample_ids.count("shrm0001") == 1
    assert [r["doc_id"] for r in collected.rows].count("shrm0001") == 1
    # 처음 뽑힌 칸(marketing × email)의 할당량에만 세므로 regulatory × email은 rega0001 하나다.
    assert collected.sample_ids == ["shrm0001", "mkea0001", "rega0001"]


def test_t10_cell_stops_at_quota(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0001", "mkea0002", "mkea0003"]}, quota=2)

    assert collected.sample_ids == ["mkea0001", "mkea0002"]
    assert "mkea0003" not in collected.by_id


def test_t11_short_cell_logs_shortfall(run_collect, caplog):
    collected = run_collect({MARKETING_EMAIL: ["mkea0001"]}, quota=2)

    assert collected.sample_ids == ["mkea0001"]
    messages = [r.getMessage() for r in caplog.records]
    assert "marketing/email: 목표 2, 수집 1, 부족 1" in messages


def test_t12_same_config_and_seed_give_same_sample(run_collect):
    cells = {
        MARKETING_EMAIL: ["mkea0001", "shrm0001", "mkea0003"],
        REGULATORY_EMAIL: ["shrm0001", "rega0001"],
        REGULATORY_DOCUMENT: ["txtd0001"],
    }
    first = run_collect(cells, seed=7, out="first")
    second = run_collect(cells, seed=7, out="second")

    assert first.sample_ids == second.sample_ids == ["mkea0001", "shrm0001", "rega0001", "txtd0001"]
    assert {r["sort"] for r in first.solr_requests + second.solr_requests} == {"random_7 asc"}


def test_t13_document_table_has_contract_columns_filled(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0002"]})

    assert collected.by_id["mkea0002"] == {
        "doc_id": "mkea0002",
        "bates": "JLI00000102",
        "title": "Age gating survey fieldwork update",
        "doc_type": ["email"],
        "topic": "Marketing;Youth Prevention;",
        "departments": ["marketing"],
        "text": (
            "Subject: Age gating survey fieldwork update\n"
            "Recruiting for the age gating survey is 60% complete.\nJLI00000102\n"
        ),
    }


def test_t14_person_metadata_is_not_stored(run_collect):
    # mkea0001: 보관자 `Morgan, Kate`, 작성자 `Kate Morgan <kate.morgan@juul.com>`
    collected = run_collect({MARKETING_EMAIL: ["mkea0001"]})

    columns = set(collected.table.column_names)
    assert not columns & {"custodian", "author", "recipient", "copied"}
    row = repr(collected.by_id["mkea0001"])
    assert "Morgan, Kate" not in row
    assert "kate.morgan@juul.com" not in row


def test_extra_collect_log_file_keeps_shortfall_and_skip_reasons(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["nobd0001", "mkea0001"]}, quota=2)

    log = "\n".join(collected.log_lines)
    assert "marketing/email: 목표 2, 수집 1, 부족 1" in log
    assert "nobd0001" in log


def test_extra_solr_query_puts_all_conditions_in_q_and_excludes_external_types(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0001"]})

    q = collected.solr_requests[0]["q"]
    assert q == (
        'collectioncode:juul AND type:email AND topic:"Marketing" AND '
        "NOT type:(publication OR article OR news OR transcript OR video OR audio)"
    )
    assert "fq" not in collected.solr_requests[0]
