"""`collect` → `index` → `eval`: 허용 부서 필드에 payload 색인을 만들어, 필터를 건 근사 검색을
바로잡는다.

테스트 이름의 `tNN`은 티켓 #49 Acceptance criteria의 사례 번호다. 로컬 모드는 payload 색인을 만들지
않으므로, 색인이 있는지는 테스트용 Qdrant 서버(`qdrant_server`)로 확인한다.
"""

from qdrant_client import models

from enterprise_rag.cli import main
from enterprise_rag.index import open_index
from tests.conftest import write_config
from tests.eval_set import EVAL_SET


def payload_schema(pipeline):
    with open_index(pipeline.config["paths"]["index"]) as client:
        return client.get_collection(pipeline.config["index"]["collection"]).payload_schema


def test_t01_index_creates_keyword_index_on_allowed_departments(server_pipeline):
    schema = payload_schema(server_pipeline)

    assert schema["departments"].data_type == models.PayloadSchemaType.KEYWORD


def add_payload_index_only(pipeline):
    return main(["--config", str(pipeline.config_path), "index", "--payload-index-only"])


def point_count(pipeline):
    with open_index(pipeline.config["paths"]["index"]) as client:
        return client.count(pipeline.config["index"]["collection"]).count


def test_t02_payload_index_is_added_to_existing_collection_without_embedding(server_pipeline):
    # payload 색인이 없던 #49 이전 색인을 흉내 낸다
    with open_index(server_pipeline.config["paths"]["index"]) as client:
        client.delete_payload_index(server_pipeline.config["index"]["collection"], "departments")
    assert "departments" not in payload_schema(server_pipeline)
    count, loads = point_count(server_pipeline), len(server_pipeline.embedder_loads)

    assert (
        main(["--config", str(server_pipeline.config_path), "index", "--payload-index-only"]) == 0
    )

    assert payload_schema(server_pipeline)["departments"].data_type == "keyword"
    assert point_count(server_pipeline) == count
    assert len(server_pipeline.embedder_loads) == loads


def without_latency(run):
    metrics = {k: v for k, v in run.metrics.items() if k != "latency_ms"}
    records = [{k: v for k, v in r.items() if k != "latency_ms"} for r in run.records]
    return metrics, records


def test_t03_adding_payload_index_keeps_eval_results(server_pipeline):
    with open_index(server_pipeline.config["paths"]["index"]) as client:
        client.delete_payload_index(server_pipeline.config["index"]["collection"], "departments")
    before = server_pipeline.eval(EVAL_SET, "t03-before")

    assert (
        main(["--config", str(server_pipeline.config_path), "index", "--payload-index-only"]) == 0
    )
    after = server_pipeline.eval(EVAL_SET, "t03-after")

    assert without_latency(after) == without_latency(before)


def test_t04_acl_filter_on_server_gives_zero_retrieval_leak_rate(server_pipeline):
    run = server_pipeline.eval(EVAL_SET, "t04", acl_filter=True)

    assert run.metrics["retrieval_leak_rate"] == 0.0


def test_extra_payload_index_only_needs_server_and_existing_collection(pipeline, qdrant_server):
    # 로컬 모드 색인(fixture 기본값)
    assert add_payload_index_only(pipeline) == 2
    # 서버에 컬렉션이 아직 없음
    pipeline.config["paths"]["index"] = qdrant_server
    write_config(pipeline.config_path, pipeline.config)
    assert add_payload_index_only(pipeline) == 2
