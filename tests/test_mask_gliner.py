"""`collect` → `index`: 이름 찾기 모델을 GLiNER로 고르면 홀로 쓰인 이름을 가리고, 역할 명사와 직함은
가리지 않는다. 보정 규칙(#19·#38)은 그대로 적용된다.

테스트 이름의 `tNN`은 티켓 #44 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
사례마다 그 문장만 본문에 든 fixture 문서(`mskc00NN`, NN은 사례 번호)를 색인해 청크를 읽는다.
GLiNER와 spaCy는 실제 모델을 쓴다.
"""

from enterprise_rag.cli import main
from tests.conftest import write_config


def indexed(make_pipeline, doc_id, name_model="gliner"):
    """문서 하나만 이름 찾기 모델을 정해 `collect` → `index`하고 색인된 청크 내용을 돌려준다."""
    pipeline = make_pipeline({("Product Design & Quality", "document"): [doc_id]})
    pipeline.config["pii"]["name_model"] = name_model
    pipeline.index()
    return pipeline.chunks(doc_id)


def test_t02_names_alone_in_table_cells_are_masked(make_pipeline):
    [text] = indexed(make_pipeline, "mskc0002")

    assert "Eddy" not in text
    assert "Carol" not in text


def test_t03_role_nouns_are_left_as_is(make_pipeline):
    assert indexed(make_pipeline, "mskc0003") == [
        "The Distributor shall notify the Supplier and each User."
    ]


def test_t04_title_before_name_is_left_and_name_is_masked(make_pipeline):
    [text] = indexed(make_pipeline, "mskc0004")

    assert "Mark Warner" not in text
    assert "Warner" not in text
    assert "Sen." in text


def test_t05_masker_rules_still_apply_to_gliner_names(make_pipeline):
    [text] = indexed(make_pipeline, "mskb0001")

    assert "Hoang" not in text


def test_t06_unknown_name_model_is_refused_naming_valid_values(pipeline, capsys):
    pipeline.config["pii"]["name_model"] = "bert"
    write_config(pipeline.config_path, pipeline.config)

    code = main(["--config", str(pipeline.config_path), "index"])

    assert code != 0
    err = capsys.readouterr().err
    assert "spacy" in err
    assert "gliner" in err


def test_t07_changing_name_model_masks_again(pipeline, caplog):
    caplog.set_level("INFO")
    pipeline.config["pii"]["name_model"] = "gliner"
    caplog.clear()

    pipeline.index()

    messages = [r.getMessage() for r in caplog.records]
    assert any("마스킹했다" in m for m in messages)
    assert not any("저장된 마스킹 결과를 썼다" in m for m in messages)
