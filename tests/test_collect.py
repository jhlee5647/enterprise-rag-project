"""`collect` 명령: 가짜 IDL 서버를 상대로 실행하고 문서 표와 문서 ID 목록을 확인한다."""

MARKETING_EMAIL = ("Marketing", "email")
MARKETING_DOCUMENT = ("Marketing", "document")
REGULATORY_EMAIL = ("Government & Public Affairs", "email")


def test_marketing_email_is_allowed_to_marketing(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0001"]})

    assert collected.by_id["mkea0001"]["departments"] == ["marketing"]


def test_document_with_two_mapped_topics_is_allowed_to_both_departments(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["shrm0001"]})

    assert collected.by_id["shrm0001"]["departments"] == ["marketing", "regulatory"]


def test_unmapped_topic_value_is_ignored(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0002"]})

    assert collected.by_id["mkea0002"]["departments"] == ["marketing"]


def test_topic_matching_ignores_case_and_trailing_separator(run_collect):
    collected = run_collect({("Product Design & Quality", "presentation"): ["prdp0001"]})

    assert collected.by_id["prdp0001"]["departments"] == ["product"]


def test_documents_without_allowed_departments_are_excluded_and_not_counted(run_collect):
    # flav0001: 토픽이 `Flavors`뿐, notp0001: `topic` 필드 없음. 할당량 2를 차지하지 않는다.
    collected = run_collect({MARKETING_EMAIL: ["flav0001", "notp0001", "mkea0001", "mkea0003"]})

    assert collected.sample_ids == ["mkea0001", "mkea0003"]
    assert set(collected.by_id) == {"mkea0001", "mkea0003"}


def test_cell_stops_at_quota(run_collect):
    collected = run_collect({MARKETING_EMAIL: ["mkea0001", "mkea0002", "mkea0003"]}, quota=2)

    assert collected.sample_ids == ["mkea0001", "mkea0002"]
    assert "mkea0003" not in collected.by_id


def test_document_picked_in_two_cells_is_stored_once(run_collect):
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
