"""`collect` → `index` → `eval`: 하이브리드 검색은 RRF 점수가 같으면 정해진 순서로 줄을 세워, 같은
질문에 같은 결과를 돌려준다.

테스트 이름의 `tNN`은 티켓 #59 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
RRF 점수는 순위 r(1부터)마다 1 ÷ (1 + r)이다(1위 0.5, 2위 0.33). T1·T2는 순위 관계를 정확히
만들려고 합치기(`ask.rrf`)에 후보 목록을 직접 넣는다. 나머지는 `eval`로 확인한다. 상위 1개를 뽑으면
후보는 4개씩이다.
- "What does Zorvex have in it?": dense 1위는 rrfb0001(기능어만 겹침, BM25 없음), BM25 1위는
  rrfy0001(`zorvex`, dense 후보 4개 밖). 둘 다 0.5점으로 동점이다.
- "What does Plumbic have in it?": rrfa0001이 dense·BM25 모두 1위(1.0점), dense 2위는
  rrfb0001(0.33점).
"""

from types import SimpleNamespace

import pytest

from enterprise_rag import ask
from tests.eval_set import row

DENSE_ONLY, BM25_ONLY, BOTH = "rrfb0001", "rrfy0001", "rrfa0001"
TIE_CELLS = {
    ("Marketing", "document"): ["rrfb0001", "rrfb0002"],
    ("Marketing", "presentation"): ["rrfb0003", "rrfb0004"],
    ("Marketing", "email"): [BM25_ONLY, BOTH],
    ("Government & Public Affairs", "document"): ["rrfc0001"],
}
TIE = "What does Zorvex have in it?"
NO_TIE = "What does Plumbic have in it?"


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


@pytest.fixture
def tie_pipeline(make_pipeline):
    return make_pipeline(TIE_CELLS)


def candidates(*ids):
    """검색 후보 목록. 순서가 순위다."""
    return [SimpleNamespace(id=n, payload={"doc_id": i}) for n, i in ids]


def fused(dense, bm25, limit):
    return [c["doc_id"] for c in ask.rrf(candidates(*dense), candidates(*bm25), limit)]


def test_t01_rank_two_tie_goes_to_the_chunk_with_a_dense_rank():
    # A: dense 1위·BM25 1위(1.0), B: dense 2위에만(1/3), Y: BM25 2위에만(1/3)
    dense = [(1, "A"), (2, "B")]
    bm25 = [(1, "A"), (3, "Y")]

    assert fused(dense, bm25, 2) == ["A", "B"]
    # 청크 ID를 바꿔도(Y가 더 작아도) dense 순위가 있는 B가 먼저다
    assert fused([(5, "A"), (9, "B")], [(5, "A"), (1, "Y")], 2) == ["A", "B"]


def test_t02_rrf_constant_is_one():
    # 상수 1: A(dense 1위) 1/2, Y(BM25 1위) 1/2, Z(dense 3위·BM25 3위) 1/4 + 1/4 = 1/2로 셋이 같고,
    # 동점 순서(더 좋은 순위 → dense 순위)로 A, Y, Z. 상수가 60이면 Z가 2/63으로 1/61보다 커서
    # 맨 앞이 된다
    dense = [(1, "A"), (2, "X"), (3, "Z")]
    bm25 = [(4, "Y"), (5, "W"), (3, "Z")]

    assert fused(dense, bm25, 3) == ["A", "Y", "Z"]


def test_extra_rank_one_tie_in_eval_goes_to_the_dense_chunk(tie_pipeline):
    run = tie_pipeline.eval(
        [row(TIE, BM25_ONLY, "marketing", "단일 홉", "O")], "t01", top_k=1, retrieval="hybrid"
    )

    assert retrieved_ids(run.record(TIE, "marketing")) == [DENSE_ONLY]


def test_extra_chunk_in_both_searches_comes_before_dense_only_chunk_in_eval(tie_pipeline):
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
    # rrfc0001(regulatory 전용)에도 `zorvex`가 있어 BM25 후보가 될 수 있다
    rows = [row(TIE, BM25_ONLY, "marketing", "단일 홉", "O")]
    run = tie_pipeline.eval(rows, "t04", top_k=5, retrieval="hybrid", acl_filter=True)

    assert "rrfc0001" not in retrieved_ids(run.record(TIE, "marketing"))
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
