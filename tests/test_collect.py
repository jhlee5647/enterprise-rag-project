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
