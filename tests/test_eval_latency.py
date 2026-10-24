"""`collect` → `index` → `eval`: 실행마다 단계별 지연 시간을 남기고 지표 JSON에 중앙값·p95를 남긴다.

테스트 이름의 `tNN`은 티켓 #48 Acceptance criteria의 사례 번호다. 시간 값 자체는 환경마다 달라
기대값으로 쓰지 않는다(T3만 정해 둔 기록으로 요약 계산을 확인한다).
"""

import time

from enterprise_rag import embedding, metrics
from tests.eval_set import Q1, Q2, row
from tests.fake_models import FakeEmbedder

STAGES = ["embed", "search", "generate", "total"]
TWO_QUESTIONS = [
    row(Q1, "inca0001", "marketing", "단일 홉", "O"),
    row(Q2, "pmtb0001", "regulatory", "단일 홉", "O"),
]


def test_t01_each_run_records_stage_latencies(pipeline):
    run = pipeline.eval(TWO_QUESTIONS, "t01")

    assert len(run.records) == 6
    for record in run.records:
        latency = record["latency_ms"]
        assert sorted(latency) == sorted(STAGES)
        assert all(latency[s] >= 0 for s in STAGES)
        assert latency["total"] >= latency["embed"] + latency["search"] + latency["generate"]


def test_t02_metrics_summarize_each_stage_with_median_and_p95(pipeline):
    run = pipeline.eval(TWO_QUESTIONS, "t02")

    summary = run.metrics["latency_ms"]
    assert sorted(summary) == sorted(STAGES)
    for stage in STAGES:
        assert sorted(summary[stage]) == ["median", "p95"]
        assert all(isinstance(v, int) for v in summary[stage].values())
        assert summary[stage]["p95"] >= summary[stage]["median"] >= 0


def test_t03_summary_interpolates_between_sorted_values():
    records = [
        {"latency_ms": {"embed": 0, "search": 0, "generate": 0, "total": t}}
        for t in [300, 100, 400, 200]
    ]

    assert metrics.latency_summary(records)["total"] == {"median": 250, "p95": 385}


def test_t04_single_run_summary_equals_its_values(pipeline):
    run = pipeline.eval([row(Q1, "inca0001", "marketing", "단일 홉", "O")], "t04")

    # 질문 하나를 부서 3개로 실행하므로, 그중 marketing 실행 하나만 남긴 기록으로 요약한다
    [record] = [r for r in run.records if r["department"] == "marketing"]
    summary = metrics.latency_summary([record])
    for stage, value in record["latency_ms"].items():
        assert summary[stage] == {"median": value, "p95": value}


def test_t05_run_without_llm_call_has_zero_generation_time(make_pipeline):
    # product 부서 문서가 없는 색인: product로 실행하면 ACL 필터 뒤 남은 청크가 없다
    pipeline = make_pipeline({("Marketing", "document"): ["inca0001", "agec0001"]})
    run = pipeline.eval([row(Q1, "inca0001", "marketing", "단일 홉", "O")], "t05")

    record = run.record(Q1, "product")
    assert record["retrieved"] == []
    assert record["latency_ms"]["generate"] == 0
    assert run.record(Q1, "marketing")["latency_ms"]["generate"] > 0


def test_t06_model_loading_is_not_counted_as_question_latency(make_pipeline, monkeypatch):
    pipeline = make_pipeline({("Marketing", "document"): ["inca0001", "agec0001"]})
    loads = []

    def slow_load(model, dtype):
        # 실제 `load_model`처럼 한 번만 불러오고, 불러오는 데 1초 걸린다
        if not loads:
            time.sleep(1)
            loads.append(FakeEmbedder())
        return loads[0]

    monkeypatch.setattr(embedding, "load_model", slow_load)
    run = pipeline.eval([row(Q1, "inca0001", "marketing", "단일 홉", "O")], "t06")

    assert all(r["latency_ms"]["embed"] < 1000 for r in run.records)


def test_t07_latency_adds_fields_without_changing_others(pipeline):
    run = pipeline.eval(TWO_QUESTIONS, "t07")

    assert sorted(run.metrics) == sorted(
        [
            "retrieval_leak_rate",
            "refusal_accuracy",
            "hit_rate",
            "pii_miss_rate",
            "pii_false_positive_rate",
            "latency_ms",
        ]
    )
    assert sorted(run.records[0]) == sorted(
        ["question", "department", "expected", "retrieved", "answer", "hit_rate", "latency_ms"]
    )
