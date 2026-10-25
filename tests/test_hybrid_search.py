"""`collect` → `index` → `eval`: 검색 방식 `hybrid`는 dense와 BM25 결과를 순위로 합쳐(RRF) 검색한다.

테스트 이름의 `tNN`은 티켓 #52 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
dense 임베딩은 가짜(단어 빈도), BM25 단어 가중치는 실제 로컬 모델(`Qdrant/bm25`)을 쓴다.
정답 R(hyba0001)은 질문과 드문 고유명 `Zorvex`만 겹친다. hybb0001·hybb0002는 질문과 기능어만 많이
겹쳐 가짜 dense에서는 R보다 가깝지만, BM25는 기능어를 빼므로 점수가 없다. hybc0001은 `Zorvex`가 든
regulatory 전용 문서다.
"""

import pytest
from qdrant_client import models

from enterprise_rag.index import open_index
from tests.eval_set import row

R, OTHER_DEPT = "hyba0001", "hybc0001"
HYBRID_CELLS = {
    ("Marketing", "document"): [R, "hybb0001"],
    ("Marketing", "presentation"): ["hybb0002"],
    ("Government & Public Affairs", "document"): [OTHER_DEPT],
}
QUESTION = "What role does Zorvex have in the launch?"
ROWS = [row(QUESTION, R, "marketing", "단일 홉", "O")]


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


@pytest.fixture
def hybrid_pipeline(make_pipeline):
    return make_pipeline(HYBRID_CELLS)


def test_t01_dense_misses_document_sharing_only_a_rare_name(hybrid_pipeline):
    run = hybrid_pipeline.eval(ROWS, "t01", top_k=1, retrieval="dense")

    record = run.record(QUESTION, "marketing")
    assert R not in retrieved_ids(record)
    assert record["hit_rate"] == 0.0


def test_t02_hybrid_finds_document_sharing_only_a_rare_name(hybrid_pipeline):
    run = hybrid_pipeline.eval(ROWS, "t02", top_k=1, retrieval="hybrid")

    record = run.record(QUESTION, "marketing")
    assert retrieved_ids(record) == [R]
    assert record["hit_rate"] == 1.0


def test_t03_hybrid_applies_acl_filter_to_both_searches(hybrid_pipeline):
    # 상위 k를 크게 해도 marketing 실행에는 Zorvex가 든 regulatory 전용 문서가 나오지 않는다
    run = hybrid_pipeline.eval(ROWS, "t03", top_k=5, retrieval="hybrid", acl_filter=True)

    assert OTHER_DEPT not in retrieved_ids(run.record(QUESTION, "marketing"))
    assert run.metrics["retrieval_leak_rate"] == 0.0


def test_t04_default_retrieval_is_dense(hybrid_pipeline):
    run = hybrid_pipeline.eval(ROWS, "t04", top_k=1)

    assert hybrid_pipeline.config["ask"]["retrieval"] == "dense"
    assert run.record(QUESTION, "marketing")["hit_rate"] == 0.0


def test_t06_unknown_retrieval_stops_before_running(hybrid_pipeline, capsys):
    capsys.readouterr()

    code = hybrid_pipeline.eval(ROWS, "t06", retrieval="bm25only", expect_code=2)

    assert code == 2
    err = capsys.readouterr().err
    assert "bm25only" in err and "dense" in err and "hybrid" in err
    assert hybrid_pipeline.vllm.requests == []


def test_t07_hybrid_on_index_without_sparse_vectors_asks_to_reindex(hybrid_pipeline, capsys):
    # #52 이전 형식(dense 벡터만) 컬렉션을 만든다
    config = hybrid_pipeline.config
    with open_index(config["paths"]["index"]) as client:
        points, _ = client.scroll(config["index"]["collection"], limit=10_000, with_vectors=True)
        client.delete_collection(config["index"]["collection"])
        client.create_collection(
            config["index"]["collection"],
            vectors_config=models.VectorParams(
                size=len(points[0].vector[""]), distance=models.Distance.COSINE
            ),
        )
        client.upsert(
            config["index"]["collection"],
            [models.PointStruct(id=p.id, vector=p.vector[""], payload=p.payload) for p in points],
        )
    capsys.readouterr()

    code = hybrid_pipeline.eval(ROWS, "t07", retrieval="hybrid", expect_code=2)

    assert code == 2
    assert "index를 다시 실행" in capsys.readouterr().err
    # ask도 같다. dense로 바꾸면 이전 형식 색인으로 그대로 동작한다
    assert hybrid_pipeline("marketing", QUESTION).code == 2
    config["ask"]["retrieval"] = "dense"
    assert hybrid_pipeline("marketing", QUESTION).code == 0


def test_t08_every_chunk_has_a_sparse_vector(hybrid_pipeline):
    config = hybrid_pipeline.config
    with open_index(config["paths"]["index"]) as client:
        points, _ = client.scroll(config["index"]["collection"], limit=10_000, with_vectors=True)

    assert points
    assert all(p.vector["bm25"].indices for p in points)


def test_extra_hybrid_works_on_qdrant_server(hybrid_pipeline, qdrant_server):
    # 로컬 모드와 서버는 IDF 계산과 RRF 합치기를 따로 구현하므로 서버에서도 확인한다
    hybrid_pipeline.config["paths"]["index"] = qdrant_server
    hybrid_pipeline.index()

    run = hybrid_pipeline.eval(ROWS, "extra", top_k=1, retrieval="hybrid")

    assert retrieved_ids(run.record(QUESTION, "marketing")) == [R]
