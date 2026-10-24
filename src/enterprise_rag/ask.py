"""질의응답기: (질문, 부서)를 받아 검색 → 리랭크 → 생성을 거쳐 답변과 인용 목록을 돌려준다."""

import re
import time
from typing import Any

import httpx

from enterprise_rag import embedding
from enterprise_rag.index import department_filter, open_index

# 근거가 없을 때와 권한이 없을 때 똑같이 돌려주는 찾지 못함 응답
NOT_FOUND = "No relevant documents were found."
IDL_URL = "https://www.industrydocuments.ucsf.edu/docs/"
# 단계별 지연 시간의 단계: 질문 임베딩, 검색(리랭크 포함), 답변 생성, 전체
LATENCY_STAGES = ["embed", "search", "generate", "total"]

SYSTEM_PROMPT = f"""\
You answer questions about internal company documents.
Use only the numbered sources given. After each sentence, write the numbers of the sources \
it relies on, like [1] or [2][3].
If the sources do not answer the question, output exactly this sentence and nothing else:
{NOT_FOUND}"""


def ask(config: dict[str, Any], department: str, question: str) -> dict[str, Any]:
    """답변(`answer`), 인용 목록(`citations`), 검색된 청크 목록(`retrieved`), 단계별 지연 시간
    (`latency_ms`: 질문 임베딩 `embed`, 검색과 리랭크 `search`, 답변 생성 `generate`, 전체 `total`,
    밀리초 정수)을 돌려준다. LLM을 부르지 않으면 답변 생성 시간은 0이다.
    """
    cfg = config["ask"]
    started = time.perf_counter()
    vector = embed_question(config, question)
    embedded = time.perf_counter()
    chunks = rerank(cfg["reranker"], search(config, department, vector))
    searched = time.perf_counter()
    retrieved = [
        {
            "doc_id": c["doc_id"],
            "bates": c["bates"],
            "chunk": c["chunk"],
            "departments": c["departments"],
            "score": c["score"],
        }
        for c in chunks
    ]
    # 남은 청크가 없으면 LLM을 부르지 않는다. LLM이 찾지 못함 응답을 내면 인용 없이 그대로 돌려준다.
    answer = generate(config["llm"], question, chunks) if chunks else NOT_FOUND
    generated = time.perf_counter()
    # 시작부터 잰 시각을 밀리초 정수로 반올림한 뒤 빼서, 단계의 합이 전체를 넘지 않게 한다
    marks = [round((t - started) * 1000) for t in (embedded, searched, generated)]
    latency_ms = {
        "embed": marks[0],
        "search": marks[1] - marks[0],
        "generate": marks[2] - marks[1] if chunks else 0,
        "total": marks[2],
    }
    found = answer != NOT_FOUND
    return {
        "answer": answer,
        "citations": citations(answer, chunks) if found else [],
        "retrieved": retrieved,
        "latency_ms": latency_ms,
    }


def load_models(config: dict[str, Any]) -> None:
    """질의응답기가 쓰는 모델을 미리 불러 둔다. 모델 불러오기가 첫 질문의 지연 시간에 섞이지 않게
    `eval`이 첫 실행 전에 부른다. LLM 서버는 따로 떠 있으므로 준비하지 않는다.

    GPU 모델은 첫 임베딩에서 준비 작업을 해 1초 넘게 걸리므로, 짧은 질문을 한 번 임베딩해 둔다.
    """
    embed_question(config, "warm-up")


def embed_question(config: dict[str, Any], question: str) -> list[float]:
    """질문을 config의 임베딩 모델로 임베딩한다."""
    return embedding.load_embedder(config).embed_query(question)


def search(config: dict[str, Any], department: str, vector: list[float]) -> list[dict[str, Any]]:
    """질문 벡터와 가장 가까운 청크 상위 k개. ACL 필터가 켜져 있으면 검색 전에 허용 부서로
    거른다."""
    cfg = config["ask"]
    if cfg["retrieval"] != "dense":
        raise ValueError(f"지원하지 않는 검색 방식: {cfg['retrieval']} (구현된 것: dense)")
    acl = department_filter(department) if cfg["acl_filter"] else None
    with open_index(config["paths"]["index"]) as client:
        points = client.query_points(
            config["index"]["collection"], query=vector, query_filter=acl, limit=cfg["top_k"]
        ).points
    return [{**p.payload, "score": p.score} for p in points]


def rerank(reranker: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """리랭커 자리. `none`은 순서를 바꾸지 않고 그대로 통과시킨다."""
    if reranker != "none":
        raise ValueError(f"지원하지 않는 리랭커: {reranker} (구현된 것: none)")
    return chunks


def generate(llm: dict[str, Any], question: str, chunks: list[dict[str, Any]]) -> str:
    """청크마다 근거 번호를 붙여 vLLM(OpenAI 호환 API)에 넘기고 답변을 받는다."""
    sources = "\n".join(f"[{n}] {c['text']}" for n, c in enumerate(chunks, 1))
    response = httpx.post(
        llm["base_url"].rstrip("/") + "/chat/completions",
        json={
            "model": llm["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
            ],
            "temperature": 0,
            "max_tokens": llm["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 thinking 모드를 끈다
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def citations(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """답변에 적힌 근거 번호의 청크를 인용으로 바꾼다. 문서마다 한 번, 처음 나온 순서대로."""
    result: dict[str, dict[str, str]] = {}
    for number in re.findall(r"\[(\d+)\]", answer):
        n = int(number)
        if not 1 <= n <= len(chunks):
            continue
        c = chunks[n - 1]
        result.setdefault(
            c["doc_id"],
            {
                "doc_id": c["doc_id"],
                "bates": c["bates"],
                "url": IDL_URL + c["doc_id"],
                "title": c["title"],
            },
        )
    return list(result.values())
