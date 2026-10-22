"""질의응답기: (질문, 부서)를 받아 검색 → 리랭크 → 생성을 거쳐 답변과 인용 목록을 돌려준다."""

import re
from typing import Any

import httpx
from qdrant_client import models

from enterprise_rag import embedding
from enterprise_rag.index import COLLECTION, open_index

# 근거가 없을 때와 권한이 없을 때 똑같이 돌려주는 찾지 못함 응답
NOT_FOUND = "No relevant documents were found."
IDL_URL = "https://www.industrydocuments.ucsf.edu/docs/"

SYSTEM_PROMPT = f"""\
You answer questions about internal company documents.
Use only the numbered sources given. After each sentence, write the numbers of the sources \
it relies on, like [1] or [2][3].
If the sources do not answer the question, output exactly this sentence and nothing else:
{NOT_FOUND}"""


def ask(config: dict[str, Any], department: str, question: str) -> dict[str, Any]:
    """답변(`answer`), 인용 목록(`citations`), 검색된 청크 목록(`retrieved`)을 돌려준다."""
    cfg = config["ask"]
    chunks = rerank(cfg["reranker"], retrieve(config, department, question))
    retrieved = [
        {"doc_id": c["doc_id"], "bates": c["bates"], "chunk": c["chunk"], "score": c["score"]}
        for c in chunks
    ]
    if not chunks:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": retrieved}

    answer = generate(config["llm"], question, chunks)
    if answer == NOT_FOUND:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": retrieved}
    return {"answer": answer, "citations": citations(answer, chunks), "retrieved": retrieved}


def retrieve(config: dict[str, Any], department: str, question: str) -> list[dict[str, Any]]:
    """질문과 가장 가까운 청크 상위 k개. ACL 필터가 켜져 있으면 검색 전에 허용 부서로 거른다."""
    cfg = config["ask"]
    if cfg["retrieval"] != "dense":
        raise ValueError(f"지원하지 않는 검색 방식: {cfg['retrieval']} (구현된 것: dense)")
    vector = embedding.load_embedder(config).embed_query(question)
    acl = None
    if cfg["acl_filter"]:
        acl = models.Filter(
            must=[
                models.FieldCondition(key="departments", match=models.MatchValue(value=department))
            ]
        )
    with open_index(config["paths"]["index"]) as client:
        points = client.query_points(
            COLLECTION, query=vector, query_filter=acl, limit=cfg["top_k"]
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
            c["doc_id"], {"doc_id": c["doc_id"], "bates": c["bates"], "url": IDL_URL + c["doc_id"]}
        )
    return list(result.values())
