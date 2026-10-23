"""`collect` → `index` → `gen-eval`: 평가셋은 두고 PII 정답셋에만 청크를 더한다.

테스트 이름의 `tNN`은 티켓 #18 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
질의응답 평가셋 생성(RAGAS)은 질문을 하나도 만들지 않는 가짜로 바꿔 끈다.
NER은 실제 spaCy 모델을 쓴다.
"""

from pathlib import Path

import pytest

from enterprise_rag import gen_eval
from tests.test_pii_eval import SIX_CHUNK_CELLS

EDITED = "NAME: Edited Name"


@pytest.fixture
def six_chunks(make_pipeline, monkeypatch):
    monkeypatch.setattr(gen_eval, "synthesize", lambda config, chunks, size: [])
    return make_pipeline(SIX_CHUNK_CELLS)


def draft_then_edit_first_row(pipeline, pii_chunks=3):
    """청크 `pii_chunks`개로 초안을 만들고 첫 행의 PII 목록을 사람이 고친 것처럼 바꾼다."""
    pipeline.gen_eval(pii_chunks=pii_chunks)
    rows = pipeline.pii_draft()
    rows[0]["pii"] = EDITED
    pipeline.write_pii_answers(rows)
    return rows


def test_t01_extending_keeps_confirmed_rows_and_edited_pii(six_chunks):
    before = draft_then_edit_first_row(six_chunks)

    rows = six_chunks.gen_eval(pii_only=True, pii_chunks=5)

    assert len(rows) == 5
    kept = [(r["doc_id"], r["chunk"], r["text"]) for r in before]
    assert [(r["doc_id"], r["chunk"], r["text"]) for r in rows if r["pii"] == EDITED] == kept[:1]
    assert set(kept) <= {(r["doc_id"], r["chunk"], r["text"]) for r in rows}


def test_t02_extending_leaves_eval_set_untouched(six_chunks):
    draft_then_edit_first_row(six_chunks)
    eval_set = six_chunks.config["paths"]["eval_set"]
    with open(eval_set, "w") as f:
        f.write("검수본")

    six_chunks.gen_eval(pii_only=True, pii_chunks=5)

    assert open(eval_set).read() == "검수본"


def test_t03_six_chunks_are_split_in_half_reproducibly(six_chunks):
    draft_then_edit_first_row(six_chunks)
    extended = six_chunks.gen_eval(pii_only=True, pii_chunks=6)
    split_of = {(r["doc_id"], r["chunk"]): r["split"] for r in extended}

    assert sorted(split_of.values()) == ["dev"] * 3 + ["test"] * 3
    path = Path(six_chunks.config["paths"]["pii_answers"])
    path.unlink()
    fresh = six_chunks.gen_eval(pii_only=True, pii_chunks=6)
    assert {(r["doc_id"], r["chunk"]): r["split"] for r in fresh} == split_of


def test_t04_old_csv_without_split_gets_split_only(six_chunks):
    six_chunks.gen_eval(pii_chunks=3)
    old = [{k: r[k] for k in ("doc_id", "chunk", "text", "pii")} for r in six_chunks.pii_draft()]
    old[0]["pii"] = EDITED
    six_chunks.write_pii_answers(old)

    rows = six_chunks.gen_eval(pii_only=True, pii_chunks=3)

    assert [{k: r[k] for k in old[0]} for r in rows] == old
    assert all(r["split"] in ("dev", "test") for r in rows)


def test_t05_same_chunk_count_again_changes_nothing(six_chunks):
    draft_then_edit_first_row(six_chunks)
    six_chunks.gen_eval(pii_only=True, pii_chunks=5)
    path = Path(six_chunks.config["paths"]["pii_answers"])
    before = path.read_bytes()

    six_chunks.gen_eval(pii_only=True, pii_chunks=5)

    assert path.read_bytes() == before


def test_t06_fewer_chunks_than_rows_is_refused(six_chunks, capsys):
    draft_then_edit_first_row(six_chunks)
    six_chunks.gen_eval(pii_only=True, pii_chunks=5)
    path = Path(six_chunks.config["paths"]["pii_answers"])
    before = path.read_bytes()
    capsys.readouterr()

    code = six_chunks.gen_eval(pii_only=True, pii_chunks=2, expect_code=2)

    assert code != 0
    err = capsys.readouterr().err
    assert "5행" in err
    assert "2개" in err
    assert path.read_bytes() == before


def test_t07_changed_chunk_text_is_refused_with_doc_id(six_chunks, capsys):
    draft_then_edit_first_row(six_chunks)
    rows = six_chunks.gen_eval(pii_only=True, pii_chunks=5)
    path = Path(six_chunks.config["paths"]["pii_answers"])
    before = path.read_bytes()
    capsys.readouterr()
    # 청크 길이를 줄이면 각 문서의 0번 청크가 앞 몇 단어로 짧아진다
    six_chunks.config["index"].update(chunk_size=4, chunk_overlap=1)

    code = six_chunks.gen_eval(pii_only=True, pii_chunks=6, expect_code=2)

    assert code != 0
    err = capsys.readouterr().err
    for r in rows:
        assert r["doc_id"] in err
    assert path.read_bytes() == before


def test_extra_changed_seed_is_refused(six_chunks, capsys):
    # 예: seed를 바꿔 확정된 행의 청크가 새로 뽑은 청크에 들지 않는 경우
    draft_then_edit_first_row(six_chunks)
    path = Path(six_chunks.config["paths"]["pii_answers"])
    before = path.read_bytes()
    capsys.readouterr()

    code = six_chunks.gen_eval(pii_only=True, pii_chunks=3, seed=7, expect_code=2)

    assert code == 2
    assert "seed" in capsys.readouterr().err
    assert path.read_bytes() == before
