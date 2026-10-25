"""`collect` → `index` → `eval`: 하이브리드 검색은 RRF 점수가 같으면 정해진 순서로 줄을 세워, 같은
질문에 같은 결과를 돌려준다.

테스트 이름의 `tNN`은 티켓 #59 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
RRF 점수는 순위 r(1부터)마다 1 ÷ (1 + r)이다(1위 0.5, 2위 0.33). 상위 1개를 뽑으면 후보는 4개씩이다.
- "What does Zorvex have in it?": dense 1위는 rrfb0001(기능어만 겹침, BM25 없음), BM25 1위는
  rrfy0001(`zorvex`, dense 후보 4개 밖). 둘 다 0.5점으로 동점이다.
- "What does Plumbic have in it?": rrfa0001이 dense·BM25 모두 1위(1.0점), dense 2위는
  rrfb0001(0.33점).
"""

import pytest

from tests.eval_set import row

DENSE_ONLY, BM25_ONLY, BOTH = "rrfb0001", "rrfy0001", "rrfa0001"
TIE_CELLS = {
    ("Marketing", "document"): ["rrfb0001", "rrfb0002"],
    ("Marketing", "presentation"): ["rrfb0003", "rrfb0004"],
    ("Marketing", "email"): [BM25_ONLY, BOTH],
}
TIE = "What does Zorvex have in it?"
NO_TIE = "What does Plumbic have in it?"


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


@pytest.fixture
def tie_pipeline(make_pipeline):
    return make_pipeline(TIE_CELLS)


def test_t01_tie_goes_to_the_chunk_with_a_dense_rank(tie_pipeline):
    run = tie_pipeline.eval(
        [row(TIE, BM25_ONLY, "marketing", "단일 홉", "O")], "t01", top_k=1, retrieval="hybrid"
    )

    assert retrieved_ids(run.record(TIE, "marketing")) == [DENSE_ONLY]


def test_t02_chunk_in_both_searches_comes_before_dense_only_chunk(tie_pipeline):
    rows = [row(NO_TIE, BOTH, "marketing", "단일 홉", "O")]
    run = tie_pipeline.eval(rows, "t02", top_k=2, retrieval="hybrid")

    record = run.record(NO_TIE, "marketing")
    # 상수 1: rrfa0001은 1/2 + 1/2 = 1.0점, rrfb0001은 1/3점
    assert retrieved_ids(record) == [BOTH, DENSE_ONLY]


def test_t03_same_eval_twice_retrieves_the_same_chunks(tie_pipeline):
    rows = [
        row(TIE, BM25_ONLY, "marketing", "단일 홉", "O"),
        row(NO_TIE, BOTH, "marketing", "단일 홉", "O"),
    ]
    first = tie_pipeline.eval(rows, "t03-a", top_k=3, retrieval="hybrid")
    second = tie_pipeline.eval(rows, "t03-b", top_k=3, retrieval="hybrid")

    assert [r["retrieved"] for r in first.records] == [r["retrieved"] for r in second.records]


def test_t04_hybrid_with_acl_filter_has_zero_leak(tie_pipeline):
    rows = [row(TIE, BM25_ONLY, "marketing", "단일 홉", "O")]
    run = tie_pipeline.eval(rows, "t04", top_k=5, retrieval="hybrid", acl_filter=True)

    assert run.metrics["retrieval_leak_rate"] == 0.0


def test_t05_tie_order_is_the_same_on_qdrant_server(tie_pipeline, qdrant_server):
    tie_pipeline.config["paths"]["index"] = qdrant_server
    tie_pipeline.index()

    for n in range(5):
        run = tie_pipeline.eval(
            [row(TIE, BM25_ONLY, "marketing", "단일 홉", "O")],
            f"t05-{n}",
            top_k=1,
            retrieval="hybrid",
        )
        assert retrieved_ids(run.record(TIE, "marketing")) == [DENSE_ONLY]
