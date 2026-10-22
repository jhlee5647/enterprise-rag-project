"""`collect` → `index` → `gen-eval` → `eval`: PII 정답셋 초안을 만들고 마스킹 지표를 잰다.

테스트 이름의 `tNN`은 티켓 #7 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
질의응답 평가셋 생성(RAGAS)은 질문을 하나도 만들지 않는 가짜로 바꿔 끈다.
NER은 실제 spaCy 모델을 쓴다.
"""

import pytest

from enterprise_rag import gen_eval
from tests.eval_set import EVAL_SET

# 문서 표를 청크로 자르면 청크가 6개 나오는 fixture: 문서 6건, 문서마다 청크 하나.
# 문서 P(pnot0001)가 들어 있다.
SIX_CHUNK_CELLS = {
    ("Marketing", "email"): ["mkea0002", "mkea0003"],
    ("Marketing", "document"): ["inca0001", "agec0001"],
    ("Marketing", "presentation"): ["pnot0001", "crep0001"],
}


# PII 정답셋 fixture: 청크 3개, 정답 PII 3개. 기본 마스킹이면 마스커가 가리는 것도 이 3개다.
K1 = "Contact Kate Morgan at kate.morgan@juul.com."
K2 = "Call (415) 555-0134 today."
K3 = "The FDA reviewed the application."


def answers(k1_pii="NAME: Kate Morgan; EMAIL: kate.morgan@juul.com"):
    return [
        {"doc_id": "K1", "chunk": "0", "text": K1, "pii": k1_pii},
        {"doc_id": "K2", "chunk": "0", "text": K2, "pii": "PHONE: (415) 555-0134"},
        {"doc_id": "K3", "chunk": "0", "text": K3, "pii": ""},
    ]


@pytest.fixture
def six_chunks(make_pipeline, monkeypatch):
    monkeypatch.setattr(gen_eval, "synthesize", lambda config, chunks, size: [])
    return make_pipeline(SIX_CHUNK_CELLS)


def test_t01_draft_has_every_chunk_when_fewer_than_pii_chunks(six_chunks):
    six_chunks.gen_eval(pii_chunks=50)

    assert len(six_chunks.pii_draft()) == 6


def test_t02_draft_has_only_pii_chunks_rows(six_chunks):
    six_chunks.gen_eval(pii_chunks=3)

    assert len(six_chunks.pii_draft()) == 3


def test_t03_draft_row_has_unmasked_text_and_found_pii(six_chunks):
    six_chunks.gen_eval(pii_chunks=50)

    [p] = [r for r in six_chunks.pii_draft() if r["doc_id"] == "pnot0001"]
    assert p["text"] == (
        "Please contact Kate Morgan at kate.morgan@juul.com or (415) 555-0134 "
        "about the influencer budget."
    )
    assert "EMAIL: kate.morgan@juul.com" in p["pii"]
    assert "PHONE: (415) 555-0134" in p["pii"]


def test_t04_default_masking_misses_nothing_and_masks_nothing_extra(pipeline):
    run = pipeline.eval([], "t04", pii_answers=answers())

    assert run.metrics["pii_miss_rate"] == 0.0
    assert run.metrics["pii_false_positive_rate"] == 0.0


def test_t05_unmasked_type_counts_as_missed(pipeline):
    pipeline.config["pii"]["types"] = ["EMAIL", "PHONE"]

    run = pipeline.eval([], "t05", pii_answers=answers())

    # 정답 3개 중 Kate Morgan 1개가 남는다: 1 ÷ 3
    assert run.metrics["pii_miss_rate"] == 0.333


def test_t06_masked_value_missing_from_answers_counts_as_false_positive(pipeline):
    run = pipeline.eval([], "t06", pii_answers=answers(k1_pii="EMAIL: kate.morgan@juul.com"))

    # 가린 3개 중 Kate Morgan 1개가 정답에 없다: 1 ÷ 3
    assert run.metrics["pii_false_positive_rate"] == 0.333


def test_t07_one_metrics_json_has_qa_and_pii_metrics(pipeline):
    run = pipeline.eval(EVAL_SET, "t07", pii_answers=answers())

    assert set(run.metrics) == {
        "retrieval_leak_rate",
        "refusal_accuracy",
        "hit_rate",
        "pii_miss_rate",
        "pii_false_positive_rate",
    }
    assert None not in run.metrics.values()


def test_extra_existing_pii_answers_are_refused_and_kept(six_chunks, capsys):
    # 예: 사람이 고쳐 둔 PII 정답셋이 있는데 gen-eval을 다시 실행한 경우
    path = six_chunks.config["paths"]["pii_answers"]
    with open(path, "w") as f:
        f.write("검수본")

    code = six_chunks.gen_eval(expect_code=2)

    assert code == 2
    assert "pii_answers.csv" in capsys.readouterr().err
    assert open(path).read() == "검수본"


def test_extra_unreadable_pii_entry_stops_eval_with_row_number(pipeline, capsys):
    # 예: 사람이 고치다가 종류 이름을 틀리게 적은 경우(3행 = K2)
    rows = answers()
    rows[1]["pii"] = "PHON: (415) 555-0134"

    code = pipeline.eval([], "bad-pii", pii_answers=rows, expect_code=2)

    assert code == 2
    assert "3행" in capsys.readouterr().err
