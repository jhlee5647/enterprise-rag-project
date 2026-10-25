"""`collect` → `index` → `eval`: 리랭커를 켜면 검색 후보를 리랭커 점수로 다시 매기고, 기준 점수 아래
청크는 근거에서 뺀다.

테스트 이름의 `tNN`은 티켓 #63 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
리랭커는 청크 본문의 글자열로 점수를 정하는 가짜(`FakeReranker`)로 바꾼다. 검색은 `dense`이고, 질문
"What does Zorvex have in it?"의 marketing dense 순위는 rrfb0001(team), rrfb0003(staff),
rrfb0002(group), rrfa0001(Plumbic), rrfb0004(board), rrfy0001(zorvex) 순이다. rrfc0001은 regulatory
전용이다.
"""

import time

import pytest

from enterprise_rag import embedding
from tests.eval_set import row
from tests.fake_models import FakeReranker

CELLS = {
    ("Marketing", "document"): ["rrfb0001", "rrfb0002"],
    ("Marketing", "presentation"): ["rrfb0003", "rrfb0004"],
    ("Marketing", "email"): ["rrfy0001", "rrfa0001"],
    ("Government & Public Affairs", "document"): ["rrfc0001"],
}
QUESTION = "What does Zorvex have in it?"
ROWS = [row(QUESTION, "rrfy0001", "marketing", "단일 홉", "O")]
BGE = "BAAI/bge-reranker-v2-m3"


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


@pytest.fixture
def rerank_pipeline(make_pipeline, monkeypatch):
    pipeline = make_pipeline(CELLS)
    pipeline.config["ask"]["retrieval"] = "dense"

    def use(scores, **kwargs):
        monkeypatch.setattr(embedding, "load_reranker", lambda name: FakeReranker(scores, **kwargs))
        pipeline.config["ask"]["reranker"] = BGE

    pipeline.use_reranker = use
    return pipeline


def test_t01_reranker_picks_a_lower_search_rank(rerank_pipeline):
    rerank_pipeline.use_reranker({"group": 0.9}, default=0.6)
    run = rerank_pipeline.eval(ROWS, "t01", top_k=1)

    assert retrieved_ids(run.record(QUESTION, "marketing")) == ["rrfb0002"]


def test_t02_chunks_below_min_score_are_dropped(rerank_pipeline):
    rerank_pipeline.use_reranker({"team": 0.8, "staff": 0.3, "group": 0.6})
    run = rerank_pipeline.eval(ROWS, "t02", top_k=2)

    assert retrieved_ids(run.record(QUESTION, "marketing")) == ["rrfb0001", "rrfb0002"]


def test_t03_min_score_boundary_keeps_half_and_drops_below(rerank_pipeline):
    rerank_pipeline.use_reranker({"team": 0.5, "staff": 0.49})
    run = rerank_pipeline.eval(ROWS, "t03", top_k=2)

    assert retrieved_ids(run.record(QUESTION, "marketing")) == ["rrfb0001"]


def test_t04_all_below_min_score_gives_not_found_without_llm(rerank_pipeline):
    rerank_pipeline.use_reranker({}, default=0.1)
    run = rerank_pipeline.eval(ROWS, "t04", top_k=2)

    record = run.record(QUESTION, "marketing")
    assert record["retrieved"] == []
    assert record["answer"] == "No relevant documents were found."
    assert record["latency_ms"]["generate"] == 0
    assert rerank_pipeline.vllm.requests == []


def test_t05_record_has_rerank_score_per_chunk(rerank_pipeline):
    rerank_pipeline.use_reranker({"team": 0.8, "group": 0.6})
    run = rerank_pipeline.eval(ROWS, "t05", top_k=2)

    scores = [c["rerank_score"] for c in run.record(QUESTION, "marketing")["retrieved"]]
    assert scores == pytest.approx([0.8, 0.6])


def test_t06_acl_filter_still_applies_before_reranking(rerank_pipeline):
    # regulatory 전용 rrfc0001에 가장 높은 점수를 줘도 marketing 실행에는 나오지 않는다
    rerank_pipeline.use_reranker({"regulators": 0.99}, default=0.6)
    run = rerank_pipeline.eval(ROWS, "t06", top_k=5, acl_filter=True)

    assert "rrfc0001" not in retrieved_ids(run.record(QUESTION, "marketing"))
    assert run.metrics["retrieval_leak_rate"] == 0.0


def test_t07_equal_rerank_scores_keep_search_order(rerank_pipeline):
    # dense 순위는 rrfb0001(team) 1위, rrfb0003(staff) 2위
    rerank_pipeline.use_reranker({"staff": 0.7, "team": 0.7})
    run = rerank_pipeline.eval(ROWS, "t07", top_k=2)

    assert retrieved_ids(run.record(QUESTION, "marketing")) == ["rrfb0001", "rrfb0003"]


def without_latency(run):
    metrics = {k: v for k, v in run.metrics.items() if k != "latency_ms"}
    records = [{k: v for k, v in r.items() if k != "latency_ms"} for r in run.records]
    return metrics, records


def test_t08_no_reranker_or_none_keeps_search_results(rerank_pipeline):
    unset = rerank_pipeline.eval(ROWS, "t08-unset", top_k=2)
    rerank_pipeline.config["ask"]["reranker"] = "none"
    none = rerank_pipeline.eval(ROWS, "t08-none", top_k=2)

    assert without_latency(unset) == without_latency(none)
    assert retrieved_ids(none.record(QUESTION, "marketing")) == ["rrfb0001", "rrfb0003"]
    assert all("rerank_score" not in c for r in none.records for c in r["retrieved"])


def test_t09_unknown_reranker_stops_before_running(rerank_pipeline, capsys):
    rerank_pipeline.config["ask"]["reranker"] = "foo-reranker"
    capsys.readouterr()

    code = rerank_pipeline.eval(ROWS, "t09", expect_code=2)

    assert code == 2
    err = capsys.readouterr().err
    assert "foo-reranker" in err and BGE in err and "none" in err
    asked = rerank_pipeline("marketing", QUESTION)
    assert asked.code == 2 and "foo-reranker" in asked.err
    assert rerank_pipeline.vllm.requests == []


def test_t10_reranker_loading_is_not_counted_as_search_latency(rerank_pipeline, monkeypatch):
    loaded = []

    def slow_load(name):
        # 실제 `load_reranker`처럼 한 번만 불러오고, 불러오는 데 1초 걸린다
        if not loaded:
            time.sleep(1)
            loaded.append(FakeReranker({}, default=0.9))
        return loaded[0]

    rerank_pipeline.config["ask"]["reranker"] = BGE
    monkeypatch.setattr(embedding, "load_reranker", slow_load)
    run = rerank_pipeline.eval(ROWS, "t10", top_k=2)

    assert all(r["latency_ms"]["search"] < 1000 for r in run.records)


def test_extra_reranker_works_with_hybrid_retrieval(rerank_pipeline):
    # hybrid 후보(RRF 합친 상위 4개)에서도 리랭커가 고른다. BM25 1위인 rrfy0001(zorvex)을 올린다
    rerank_pipeline.use_reranker({"zorvex": 0.9}, default=0.2)
    run = rerank_pipeline.eval(ROWS, "extra", top_k=1, retrieval="hybrid")

    record = run.record(QUESTION, "marketing")
    assert retrieved_ids(record) == ["rrfy0001"]
    assert record["hit_rate"] == 1.0
