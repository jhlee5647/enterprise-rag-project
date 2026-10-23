"""`collect` → `index` → `eval`: 정답 문서의 사본도 적중으로 세고, 지표를 소수 셋째 자리로 맞춘다.

테스트 이름의 `tNN`은 티켓 #23 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
문서 A는 copa0001(marketing), A의 사본 A′는 copa0002(marketing, regulatory),
A와 한 글자 다른 문서 B는 copb0001(marketing)이다. 색인에는 이 세 문서만 들어간다.
"""

import pytest

from tests.eval_set import EVAL_SET, row

A, A_COPY, B = "copa0001", "copa0002", "copb0001"
COPY_CELLS = {
    ("Marketing", "document"): [A, B],
    ("Marketing", "presentation"): [A_COPY],
}
# A, A′, B에 모두 있는 단어만 쓴 질문. 정답은 A
LAUNCH = "summer pod launch budget"
# B에만 있는 `color`가 든 질문. 상위 1개로 검색하면 B가 먼저 나온다
COLOR = "color sample budget"


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


@pytest.fixture
def copies(make_pipeline):
    return make_pipeline(COPY_CELLS)


def test_t01_copy_of_answer_document_counts_as_hit(copies):
    # marketing은 A와 A′를 모두 보고 둘의 점수가 같아, A′만 검색되는 실행은 regulatory로 만든다
    run = copies.eval([row(LAUNCH, A, "marketing", "단일 홉", "O")], "t01", top_k=1)

    record = run.record(LAUNCH, "regulatory")
    assert retrieved_ids(record) == [A_COPY]
    assert record["hit_rate"] == 1.0


def test_t02_document_one_letter_different_is_not_a_copy(copies):
    run = copies.eval([row(COLOR, A, "marketing", "단일 홉", "O")], "t02", top_k=1)

    record = run.record(COLOR, "marketing")
    assert retrieved_ids(record) == [B]
    assert record["hit_rate"] == 0.0


def test_t03_copy_retrieved_by_department_it_does_not_allow_is_a_leak(copies):
    run = copies.eval(
        [row(LAUNCH, A, "marketing", "단일 홉", "O")], "t03", acl_filter=False, top_k=3
    )

    assert {"doc_id": A_COPY, "departments": ["marketing", "regulatory"]} in run.record(
        LAUNCH, "product"
    )["retrieved"]
    # 검색된 9개 중 누출: regulatory의 A, B(2개)와 product의 A, A′, B(3개). 5 ÷ 9
    # regulatory의 A는 사본 A′가 regulatory를 허용해도 문서 ID별로 누출이다
    assert run.metrics["retrieval_leak_rate"] == 0.556


def test_t04_hit_rate_is_rounded_to_three_decimals(copies):
    # 정답을 찾아야 하는 실행 3번: COLOR(정답 A)를 marketing(B만 검색, 0)과 regulatory(A′, 1)로,
    # COLOR(정답 B)를 marketing(B, 1)으로. regulatory는 B와 그 사본을 볼 수 없다. 2 ÷ 3
    rows = [
        row(COLOR, A, "marketing", "단일 홉", "O"),
        row(COLOR + " for B", B, "marketing", "단일 홉", "O"),
    ]

    run = copies.eval(rows, "t04", top_k=1)

    assert run.metrics["hit_rate"] == 0.667


def test_t05_every_ratio_in_metrics_has_at_most_three_decimals(pipeline):
    run = pipeline.eval(EVAL_SET, "t05")

    for name, value in run.metrics.items():
        assert value is None or round(value, 3) == value, name


def test_t06_department_allowed_only_on_copy_has_access(copies):
    run = copies.eval([row(LAUNCH, A, "marketing", "단일 홉", "O")], "t06", top_k=1)

    assert run.record(LAUNCH, "regulatory")["hit_rate"] is not None
    # 권한 없는 실행은 product 하나뿐이고 찾지 못함 응답을 돌려준다.
    # regulatory가 셈에 들어가면 답을 했으므로 1 ÷ 2 = 0.5가 된다
    assert run.metrics["refusal_accuracy"] == 1.0


def test_t07_department_allowed_on_neither_answer_nor_copy_has_no_access(copies):
    run = copies.eval([row(LAUNCH, A, "marketing", "단일 홉", "O")], "t07", top_k=1)

    assert run.record(LAUNCH, "product")["hit_rate"] is None
