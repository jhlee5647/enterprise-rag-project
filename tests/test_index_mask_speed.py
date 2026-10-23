"""`collect` → `index`: 마스킹을 작업 프로세스로 나눠 하고, 조건이 같으면 저장한 마스킹 결과를
다시 쓴다.

테스트 이름의 `tNN`은 티켓 #20 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
문서 P는 pnot0001이다. NER은 실제 spaCy 모델, 임베딩은 가짜다.
작업 프로세스는 띄울 때마다 spaCy 모델을 새로 불러와 느리므로, 병렬 결과를 비교하는 T1만 2개로
하고 나머지는 1개(이 프로세스에서 마스킹)로 한다. 마스커를 불렀는지는 `index`가 남기는 로그로
확인한다.
"""

from pathlib import Path

import pytest

from enterprise_rag import mask
from enterprise_rag.cli import main
from tests.conftest import PIPELINE_CELLS, write_config

P_CHUNK = "Please contact [NAME] at [EMAIL] or [PHONE] about the influencer budget."
MASKED = "마스킹했다"
REUSED = "저장된 마스킹 결과를 썼다"


def all_chunks(pipeline):
    return {d["doc_id"]: pipeline.chunks(d["doc_id"]) for d in pipeline.documents()}


def reindex(pipeline, caplog):
    """`index`를 다시 실행하고 그 실행의 로그 메시지를 돌려준다."""
    caplog.clear()
    pipeline.index()
    return [r.getMessage() for r in caplog.records]


def masked_again(messages):
    return any(MASKED in m and REUSED not in m for m in messages)


def reused(messages):
    return any(REUSED in m for m in messages)


@pytest.fixture
def indexed(pipeline, caplog):
    """작업 프로세스 1개로 한 번 색인한 `Pipeline`. 두 번째 `index`부터 로그를 본다."""
    caplog.set_level("INFO")
    return pipeline


def test_t01_two_workers_give_same_chunks_as_one(indexed, caplog):
    one = all_chunks(indexed)
    Path(indexed.config["paths"]["masked"]).unlink()
    indexed.config["pii"]["workers"] = 2

    messages = reindex(indexed, caplog)

    assert any("작업 프로세스 2개" in m for m in messages)
    assert all_chunks(indexed) == one
    assert one["pnot0001"] == [P_CHUNK]


def test_t02_same_config_reuses_saved_masking(indexed, caplog):
    first = all_chunks(indexed)

    messages = reindex(indexed, caplog)

    assert reused(messages)
    assert not masked_again(messages)
    assert all_chunks(indexed) == first


def test_t03_chunk_size_change_does_not_mask_again(indexed, caplog):
    indexed.config["index"]["chunk_size"] = 256

    messages = reindex(indexed, caplog)

    assert reused(messages)
    assert not masked_again(messages)


def test_t04_pii_types_change_masks_again(indexed, caplog):
    indexed.config["pii"]["types"] = ["EMAIL", "PHONE"]

    messages = reindex(indexed, caplog)

    assert masked_again(messages)
    assert "Kate Morgan" in indexed.chunks("pnot0001")[0]


def test_t05_changed_document_table_masks_again(indexed, run_collect, caplog):
    cells = {k: list(v) for k, v in PIPELINE_CELLS.items()}
    cells[("Government & Public Affairs", "email")].append("rega0001")
    run_collect(cells, quota=3, out="out")

    messages = reindex(indexed, caplog)

    assert masked_again(messages)


def test_t06_masker_version_bump_masks_again(indexed, caplog, monkeypatch):
    monkeypatch.setattr(mask, "VERSION", mask.VERSION + 1)

    messages = reindex(indexed, caplog)

    assert masked_again(messages)


def test_t07_zero_workers_is_refused_naming_the_setting(indexed, capsys):
    indexed.config["pii"]["workers"] = 0
    write_config(indexed.config_path, indexed.config)

    code = main(["--config", str(indexed.config_path), "index"])

    assert code != 0
    assert "pii.workers" in capsys.readouterr().err


def test_t08_no_unmapped_entity_warning_in_index_log(indexed, caplog):
    # 작업 프로세스의 로그는 caplog에 잡히지 않으므로, 마스킹을 이 프로세스에서 하는 1개로 본다
    Path(indexed.config["paths"]["masked"]).unlink()

    messages = reindex(indexed, caplog)

    assert masked_again(messages)
    assert not any("is not mapped to a Presidio entity" in m for m in messages)


def test_extra_unknown_pii_type_fails_before_starting_workers(indexed):
    # 예: config에 PII 종류를 잘못 적은 경우. 작업 프로세스 안에서 실패하면 멈춰 기다릴 수 있다
    indexed.config["pii"]["types"] = ["NAME", "SSN"]
    indexed.config["pii"]["workers"] = 2
    write_config(indexed.config_path, indexed.config)

    with pytest.raises(ValueError, match="SSN"):
        main(["--config", str(indexed.config_path), "index"])
