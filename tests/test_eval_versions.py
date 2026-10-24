"""`collect` → `index` → `eval`: 평가셋에 적은 정답 문서의 다른 판본도 적중·권한으로 센다.

테스트 이름의 `tNN`은 티켓 #50 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
문서 A는 vera0001(marketing), A의 다른 판본 A2는 vera0002(marketing, regulatory, 마지막 문장만
다름), 관계없는 문서 B는 verb0001(marketing), 다중 홉 사례의 두 번째 정답 C는
verc0001(regulatory)다.
"""

from pathlib import Path

import pytest

from enterprise_rag import gen_eval
from tests.eval_set import EVAL_SET, row
from tests.test_gen_eval import FakeRagas

A, A2, B, C = "vera0001", "vera0002", "verb0001", "verc0001"
VERSION_CELLS = {
    ("Marketing", "document"): [A, B],
    ("Marketing", "presentation"): [A2],
    ("Government & Public Affairs", "document"): [C],
}
# A와 A2에 모두 있는 단어만 쓴 질문. 정답은 A
BOOTH = "trade show booth budget spring retail launch"
# A2에만 있는 `draft`가 든 질문. marketing은 A와 A2를 모두 보지만 상위 1개로 검색하면 A2가 나온다
DRAFT = "trade show booth budget draft"


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


@pytest.fixture
def version_pipeline(make_pipeline):
    return make_pipeline(VERSION_CELLS)


@pytest.fixture
def ragas(monkeypatch):
    monkeypatch.setattr(gen_eval, "synthesize", FakeRagas())


def test_t01_other_version_counts_as_hit(version_pipeline):
    # regulatory는 A를 볼 수 없어 A2만 검색된다
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{A}={A2}")]
    run = version_pipeline.eval(rows, "t01", top_k=1)

    record = run.record(BOOTH, "regulatory")
    assert retrieved_ids(record) == [A2]
    assert record["hit_rate"] == 1.0


def test_t02_version_not_written_is_not_a_hit(version_pipeline):
    # regulatory는 A를 볼 수 없어 판본을 적지 않으면 권한 밖이 되므로, A를 보는 marketing으로 본다
    rows = [row(DRAFT, A, "marketing", "단일 홉", "O", other_versions="")]
    run = version_pipeline.eval(rows, "t02", top_k=1)

    record = run.record(DRAFT, "marketing")
    assert retrieved_ids(record) == [A2]
    assert record["hit_rate"] == 0.0


def test_t03_department_allowed_only_on_other_version_has_access(version_pipeline):
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{A}={A2}")]
    run = version_pipeline.eval(rows, "t03", top_k=1)

    assert run.record(BOOTH, "regulatory")["hit_rate"] is not None
    # 권한 없는 실행은 product 하나뿐이고 찾지 못함 응답을 돌려준다.
    # regulatory가 권한 밖으로 세지면 답을 했으므로 1 ÷ 2 = 0.5가 된다
    assert run.metrics["refusal_accuracy"] == 1.0


def test_t04_other_version_retrieved_by_department_it_does_not_allow_is_a_leak(version_pipeline):
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{A}={A2}")]
    run = version_pipeline.eval(rows, "t04", acl_filter=False, top_k=2)

    assert retrieved_ids(run.record(BOOTH, "product")) == [A, A2]
    # 검색된 6개 중 누출: regulatory의 A(1개)와 product의 A, A2(2개). 3 ÷ 6
    assert run.metrics["retrieval_leak_rate"] == 0.5


def test_t05_record_lists_other_versions_with_allowed_departments(version_pipeline):
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{A}={A2}")]
    run = version_pipeline.eval(rows, "t05", top_k=1)

    [expected] = run.record(BOOTH, "regulatory")["expected"]
    assert expected["other_versions"] == [
        {"doc_id": A2, "departments": ["marketing", "regulatory"]}
    ]


def without_latency(run):
    return {k: v for k, v in run.metrics.items() if k != "latency_ms"}


def test_t06_eval_set_without_other_versions_column_works_as_before(pipeline):
    old_columns = ["question", "answer", "doc_ids", "department", "question_type", "keep"]
    old = pipeline.eval(EVAL_SET, "t06-old", columns=old_columns)
    new = pipeline.eval(EVAL_SET, "t06-new")

    assert without_latency(old) == without_latency(new)
    assert all(e["other_versions"] == [] for r in old.records for e in r["expected"])


def test_t07_unknown_other_version_stops_before_running(version_pipeline, capsys):
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{A}=zzzz9999")]
    capsys.readouterr()

    code = version_pipeline.eval(rows, "t07", expect_code=2)

    assert code == 2
    assert "zzzz9999" in capsys.readouterr().err
    assert not (Path(version_pipeline.config["paths"]["runs"]) / "t07").exists()
    assert version_pipeline.vllm.requests == []


def test_t08_version_on_document_not_in_doc_ids_stops(version_pipeline, capsys):
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{B}={A2}")]
    capsys.readouterr()

    code = version_pipeline.eval(rows, "t08", expect_code=2)

    assert code == 2
    assert B in capsys.readouterr().err
    assert not (Path(version_pipeline.config["paths"]["runs"]) / "t08").exists()
    assert version_pipeline.vllm.requests == []


def test_extra_same_answer_document_written_twice_stops(version_pipeline, capsys):
    # 예: 사람이 CSV를 고치다 같은 정답 문서의 판본 목록을 두 번 적은 경우
    rows = [row(BOOTH, A, "marketing", "단일 홉", "O", other_versions=f"{A}={A2};{A}={B}")]
    capsys.readouterr()

    code = version_pipeline.eval(rows, "extra", expect_code=2)

    assert code == 2
    assert A in capsys.readouterr().err


def test_t09_multi_hop_counts_version_for_one_answer_document(version_pipeline):
    question = "spring retail launch"
    rows = [
        row(question, f"{A};{C}", "regulatory", "다중 홉 구체형", "O", other_versions=f"{A}={A2}")
    ]
    run = version_pipeline.eval(rows, "t09", top_k=2)

    record = run.record(question, "regulatory")
    assert sorted(retrieved_ids(record)) == [A2, C]
    assert record["hit_rate"] == 1.0


def test_t10_eval_set_draft_has_empty_other_versions_column(pipeline, ragas):
    rows = pipeline.gen_eval(eval_only=True)

    assert rows
    assert all(r["other_versions"] == "" for r in rows)
