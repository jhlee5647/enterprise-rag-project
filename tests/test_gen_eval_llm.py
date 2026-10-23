"""`collect` → `index` → `gen-eval`: 평가셋 질문을 만드는 생성 LLM을 답변 LLM과 따로 지정한다.

테스트 이름의 `tNN`은 티켓 #15 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
RAGAS는 가짜로 바꾸지 않고 가짜 vLLM 서버 두 개(생성 LLM A, 답변 LLM B)로 실제로 돌린다. 가짜
서버는 RAGAS 프롬프트에 빈 JSON을 돌려주므로 질문은 만들어지지 않고, 요청이 간 곳만 확인한다.
"""

from tests.fake_models import FakeVLLM


def models_asked(server):
    return {r["model"] for r in server.requests}


def test_t01_generation_requests_go_only_to_configured_generation_llm(pipeline):
    with FakeVLLM() as generator:
        pipeline.config["gen_eval"]["llm"] = {
            "model": "gen-model",
            "base_url": generator.base_url,
            "max_tokens": 512,
        }

        pipeline.gen_eval(eval_only=True)

    assert generator.requests
    assert models_asked(generator) == {"gen-model"}
    # LangChain은 출력 길이 상한을 OpenAI의 새 이름(max_completion_tokens)으로 보낸다
    assert {r.get("max_completion_tokens", r.get("max_tokens")) for r in generator.requests} == {
        512
    }
    assert pipeline.vllm.requests == []


def test_t02_without_generation_llm_requests_go_to_answer_llm(pipeline):
    pipeline.gen_eval(eval_only=True)

    assert pipeline.vllm.requests
    assert models_asked(pipeline.vllm) == {"fake-llm"}
