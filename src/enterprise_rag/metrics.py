"""지표 계산: 실행 기록으로 검색 누출률, 거부 정확도, 검색 적중률을, PII 정답셋 청크의 마스킹
결과로 PII 누락률, PII 오탐률을 낸다.

실행 기록 하나는 (질문, 부서) 한 번의 실행이다. `department`, `retrieved`(검색된 청크의
`doc_id`와 허용 부서 `departments`), `answer`, `expected`(정답 문서의 `doc_id`와 허용 부서)가
들어 있다.

PII 청크 하나에는 `answers`(사람이 확정한 정답 PII 값), `masked_text`(마스킹한 글),
`masked`(마스커가 가린 원래 값)가 들어 있다. 비교는 위치가 아니라 값으로 한다.
"""

from typing import Any

from enterprise_rag.ask import NOT_FOUND


def retrieval_leak_rate(records: list[dict[str, Any]]) -> float | None:
    """모든 실행의 검색된 청크 중 실행한 부서가 허용 부서에 없는 청크의 비율. 청크가 없으면 None."""
    chunks = [(r["department"], c) for r in records for c in r["retrieved"]]
    if not chunks:
        return None
    leaked = sum(department not in c["departments"] for department, c in chunks)
    return leaked / len(chunks)


def refusal_accuracy(records: list[dict[str, Any]]) -> float | None:
    """권한 밖 실행 중 찾지 못함 응답을 돌려준 비율. 권한 밖 실행이 없으면 None."""
    denied = [r for r in records if not has_access(r)]
    if not denied:
        return None
    return sum(r["answer"] == NOT_FOUND for r in denied) / len(denied)


def has_access(record: dict[str, Any]) -> bool:
    """정답 문서 중 하나 이상의 허용 부서에 실행한 부서가 들어 있으면 권한이 있다."""
    return any(record["department"] in d["departments"] for d in record["expected"])


def hit_rate(record: dict[str, Any]) -> float | None:
    """실행 한 번의 검색 적중률: 정답 문서 중 검색된 비율. 권한이 없는 실행이면 None.

    다중 홉 질문은 정답 2개 중 1개를 찾으면 0.5다.
    """
    if not has_access(record):
        return None
    expected = {d["doc_id"] for d in record["expected"]}
    retrieved = {c["doc_id"] for c in record["retrieved"]}
    return len(expected & retrieved) / len(expected)


def mean_hit_rate(records: list[dict[str, Any]]) -> float | None:
    """권한이 있는 실행들의 검색 적중률 평균. 그런 실행이 없으면 None."""
    rates = [rate for r in records if (rate := hit_rate(r)) is not None]
    return sum(rates) / len(rates) if rates else None


def pii_miss_rate(chunks: list[dict[str, Any]]) -> float | None:
    """정답 PII 중 마스킹 후에도 원래 값이 그대로 남은 비율. 정답 PII가 없으면 None.

    예: 정답 3개 중 1개가 남으면 1 ÷ 3 = 0.333. 소수 셋째 자리로 반올림한다.
    """
    answers = [(c, value) for c in chunks for value in c["answers"]]
    if not answers:
        return None
    missed = sum(value in c["masked_text"] for c, value in answers)
    return round(missed / len(answers), 3)


def pii_false_positive_rate(chunks: list[dict[str, Any]]) -> float | None:
    """마스커가 가린 것 중 정답 PII에 없는 값의 비율. 가린 것이 없으면 None.

    예: 3개를 가렸는데 1개가 정답에 없으면 1 ÷ 3 = 0.333. 소수 셋째 자리로 반올림한다.
    """
    masked = [(c, value) for c in chunks for value in c["masked"]]
    if not masked:
        return None
    wrong = sum(value not in c["answers"] for c, value in masked)
    return round(wrong / len(masked), 3)
