"""`collect` → `index` → `eval`: 평가셋으로 권한·검색 지표를 가짜 임베딩과 가짜 LLM으로 확인한다.

테스트 이름의 `tNN`은 티켓 #5 Acceptance criteria의 사례 번호다.
문서 A는 inca0001, B는 pmtb0001, C는 agec0001, D는 crep0001이다.
"""

from tests.eval_set import EVAL_SET, Q1, Q2, Q3

NOT_FOUND = "No relevant documents were found."


def retrieved_ids(record):
    return [c["doc_id"] for c in record["retrieved"]]


def test_t01_each_kept_question_runs_once_per_department(pipeline):
    run = pipeline.eval(EVAL_SET, "t01")

    assert len(run.records) == 9
    assert {(r["question"], r["department"]) for r in run.records} == {
        (q, d) for q in (Q1, Q2, Q3) for d in ("marketing", "regulatory", "product")
    }


def test_t02_acl_filter_on_gives_zero_retrieval_leak_rate(pipeline):
    run = pipeline.eval(EVAL_SET, "t02", acl_filter=True)

    assert run.metrics["retrieval_leak_rate"] == 0.0


def test_t03_acl_filter_off_gives_positive_retrieval_leak_rate(pipeline):
    run = pipeline.eval(EVAL_SET, "t03", acl_filter=False)

    assert run.metrics["retrieval_leak_rate"] > 0.0
    # 예: Q2를 marketing으로 실행하면 regulatory 전용 문서 B의 청크가 검색된다
    retrieved = run.record(Q2, "marketing")["retrieved"]
    assert {"doc_id": "pmtb0001", "departments": ["regulatory"]} in retrieved


def test_t04_out_of_permission_runs_get_not_found_and_refusal_accuracy_is_one(pipeline):
    run = pipeline.eval(EVAL_SET, "t04", acl_filter=True, top_k=5)

    assert run.record(Q2, "marketing")["answer"] == NOT_FOUND
    assert run.record(Q2, "product")["answer"] == NOT_FOUND
    assert run.metrics["refusal_accuracy"] == 1.0


def test_t05_single_answer_question_found_in_top_1_has_hit_rate_one(pipeline):
    run = pipeline.eval(EVAL_SET, "t05", acl_filter=True, top_k=1)

    record = run.record(Q1, "marketing")
    assert retrieved_ids(record) == ["inca0001"]
    assert record["hit_rate"] == 1.0


def test_t06_multi_hop_question_with_one_of_two_answers_found_has_hit_rate_half(pipeline):
    run = pipeline.eval(EVAL_SET, "t06", acl_filter=True, top_k=1)

    record = run.record(Q3, "marketing")
    [found] = retrieved_ids(record)
    assert found in ("inca0001", "crep0001")
    assert record["hit_rate"] == 0.5


def test_t07_run_folder_has_config_copy_metrics_and_records(pipeline):
    run = pipeline.eval(EVAL_SET, "test-a")

    assert run.folder.name == "test-a"
    assert run.config_copy.read_text() == pipeline.config_path.read_text()
    assert set(run.metrics) >= {"retrieval_leak_rate", "refusal_accuracy", "hit_rate"}
    assert len(run.records) == 9


def test_extra_existing_run_name_is_refused_and_kept(pipeline, capsys):
    first = pipeline.eval(EVAL_SET, "same")
    before = (first.folder / "metrics.json").read_text()

    code = pipeline.eval(EVAL_SET, "same", acl_filter=False, expect_code=2)

    assert code == 2
    assert "same" in capsys.readouterr().err
    assert (first.folder / "metrics.json").read_text() == before
