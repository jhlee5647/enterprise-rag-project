"""`collect` → `index` → `gen-eval --eval-only`: 확정한 PII 정답셋은 두고 평가셋만 만든다.

테스트 이름의 `tNN`은 티켓 #25 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
RAGAS 호출은 유형마다 질문 하나씩을 돌려주는 가짜로 바꾼다.
"""

from pathlib import Path

import pytest

from enterprise_rag import gen_eval
from tests.test_gen_eval import FakeRagas


@pytest.fixture
def ragas(monkeypatch):
    monkeypatch.setattr(gen_eval, "synthesize", FakeRagas())


def test_t01_eval_set_is_made_and_confirmed_pii_answers_are_untouched(pipeline, ragas):
    pii_answers = Path(pipeline.config["paths"]["pii_answers"])
    pii_answers.write_text("검수본")

    rows = pipeline.gen_eval(eval_only=True)

    assert {r["department"] for r in rows} == {"marketing", "regulatory", "product"}
    assert pii_answers.read_text() == "검수본"


def test_t02_no_pii_answers_csv_is_made(pipeline, ragas):
    pipeline.gen_eval(eval_only=True)

    assert Path(pipeline.config["paths"]["eval_set"]).exists()
    assert not Path(pipeline.config["paths"]["pii_answers"]).exists()


def test_t03_existing_eval_set_is_refused_and_kept(pipeline, ragas, capsys):
    eval_set = Path(pipeline.config["paths"]["eval_set"])
    eval_set.write_text("검수본")

    code = pipeline.gen_eval(eval_only=True, expect_code=2)

    assert code == 2
    assert "eval_set.csv" in capsys.readouterr().err
    assert eval_set.read_text() == "검수본"
