"""`collect` → `index` → `ask`: 임베딩을 설정한 정밀도로 하고, 청크를 묶어서 임베딩·저장한다.

테스트 이름의 `tNN`은 티켓 #21 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
임베딩은 가짜이고, 정밀도는 가짜 임베딩 모델을 불러올 때 받은 값으로 확인한다.
"""

from enterprise_rag import index
from enterprise_rag.cli import main
from enterprise_rag.index import open_index
from tests.conftest import write_config


def points(pipeline):
    """색인된 점마다 (ID, 부가 정보, 벡터), ID 순서대로."""
    with open_index(pipeline.config["paths"]["index"]) as client:
        found, _ = client.scroll(
            pipeline.config["index"]["collection"], limit=10_000, with_vectors=True
        )
    return sorted((p.id, p.payload, p.vector) for p in found)


def test_t01_batched_index_has_same_points_as_one_batch(pipeline, monkeypatch):
    monkeypatch.setattr(index, "EMBED_BATCH", 10_000)
    monkeypatch.setattr(index, "UPSERT_BATCH", 10_000)
    pipeline.index()
    one_batch = points(pipeline)
    # 질의응답 fixture는 청크가 열 몇 개라, 묶음을 작게 해 여러 묶음과 나머지 묶음을 거치게 한다
    monkeypatch.setattr(index, "EMBED_BATCH", 5)
    monkeypatch.setattr(index, "UPSERT_BATCH", 3)

    pipeline.index()

    batched = points(pipeline)
    assert [i for i, _, _ in batched] == list(range(len(one_batch)))
    assert batched == one_batch
    assert len(batched) > 5


def test_t02_documents_and_questions_are_embedded_in_bfloat16_by_default(pipeline):
    pipeline.embedder_loads.clear()
    pipeline.index()
    assert pipeline.embedder_loads == ["bfloat16"]

    pipeline.embedder_loads.clear()
    pipeline("marketing", "influencer budget")
    assert pipeline.embedder_loads == ["bfloat16"]


def test_t03_float32_setting_is_passed_to_the_model(pipeline):
    pipeline.config["embedding"]["dtype"] = "float32"
    pipeline.embedder_loads.clear()

    pipeline.index()

    assert pipeline.embedder_loads == ["float32"]


def test_t04_unknown_dtype_is_refused_naming_the_valid_values(pipeline, capsys):
    pipeline.config["embedding"]["dtype"] = "float8"
    write_config(pipeline.config_path, pipeline.config)

    code = main(["--config", str(pipeline.config_path), "index"])

    assert code != 0
    err = capsys.readouterr().err
    assert "bfloat16" in err
    assert "float32" in err
