"""질의응답기: (질문, 부서)를 받아 검색 → 리랭크 → 생성을 거쳐 답변과 인용 목록을 돌려준다."""

import re
import time
from fractions import Fraction
from typing import Any

import httpx
from qdrant_client import models

from enterprise_rag import embedding
from enterprise_rag.index import SPARSE_VECTOR, department_filter, open_index


class AskError(Exception):
    """질의응답을 시작할 수 없을 때(설정이나 색인 형식 문제). 메시지를 그대로 보여 준다."""


# 근거가 없을 때와 권한이 없을 때 똑같이 돌려주는 찾지 못함 응답
NOT_FOUND = "No relevant documents were found."
IDL_URL = "https://www.industrydocuments.ucsf.edu/docs/"
# 검색 방식: dense(질문 임베딩과 가까운 청크), hybrid(dense와 BM25를 RRF로 합침, #52)
RETRIEVALS = ["dense", "hybrid"]
# config에 검색 방식이 없을 때 쓰는 값
DEFAULT_RETRIEVAL = "dense"
# 하이브리드 검색에서 dense·BM25 각각 가져올 후보 수는 상위 k의 이 배수다
HYBRID_CANDIDATES = 4
# RRF 점수는 순위 r(1부터)마다 1 ÷ (RRF_CONSTANT + r)을 더한다. #52의 Qdrant 기본값과 같은 값이다
# (Qdrant는 0부터 센 순위에 k=2). 값을 바꾸면 결과가 바뀌므로 측정으로 정한다 (#59)
RRF_CONSTANT = 1
# 리랭커: none(그대로 통과) 또는 로컬 리랭커 모델 이름
# (#63, `docs/design/answer-improvement.md` 3절)
RERANKERS = ["none", "Qwen/Qwen3-Reranker-0.6B", "BAAI/bge-reranker-v2-m3"]
# 리랭커를 켜면 검색이 상위 k의 이 배수를 후보로 가져오고, 리랭커가 그중 상위 k개를 고른다
RERANK_CANDIDATES = 4
# 리랭커 점수(0~1 확률)가 이보다 낮은 청크는 근거에서 뺀다. 측정으로 고르지 않은 일반적인 기준
DEFAULT_RERANK_MIN_SCORE = 0.5
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
    started = time.perf_counter()
    vector = embed_question(config, question)
    embedded = time.perf_counter()
    chunks = rerank(config, question, search(config, department, question, vector))
    searched = time.perf_counter()
    retrieved = [
        {
            "doc_id": c["doc_id"],
            "bates": c["bates"],
            "chunk": c["chunk"],
            "departments": c["departments"],
            "score": c["score"],
            **({"rerank_score": c["rerank_score"]} if "rerank_score" in c else {}),
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
    hybrid이면 BM25 단어 가중치 모델도, 리랭커를 켰으면 리랭커도 한 번 써 둔다.
    """
    embed_question(config, "warm-up")
    if retrieval_of(config) == "hybrid":
        embedding.bm25_query("warm-up")
    if reranker_of(config) != "none":
        embedding.load_reranker(reranker_of(config)).predict([("warm-up", "warm-up")])


def embed_question(config: dict[str, Any], question: str) -> list[float]:
    """질문을 config의 임베딩 모델로 임베딩한다."""
    return embedding.load_embedder(config).embed_query(question)


def search(
    config: dict[str, Any], department: str, question: str, vector: list[float]
) -> list[dict[str, Any]]:
    """질문과 가장 가까운 청크 상위 k개(리랭커를 켜면 리랭커가 고를 후보로 상위 k의 4배). ACL
    필터가 켜져 있으면 검색 전에 허용 부서로 거른다.

    dense는 질문 벡터로 정확한 최근접 검색을 한다. hybrid는 dense와 BM25(질문의 단어 가중치로
    sparse 벡터 검색)가 각각 후보를 가져오고 `rrf`로 합친다. ACL 필터는 두 검색 모두에 건다.
    """
    cfg = config["ask"]
    retrieval = retrieval_of(config)
    check_retrieval(retrieval)
    limit = cfg["top_k"] * (RERANK_CANDIDATES if reranker_of(config) != "none" else 1)
    acl = department_filter(department) if cfg["acl_filter"] else None
    # 근사 검색(HNSW)은 상위 5개가 정확한 검색과 자주 달라 비교를 흔들었고, 허용 부서
    # payload 색인으로도 고쳐지지 않았다. 그래서 정확한 최근접 검색을 한다
    # (`docs/design/retrieval-improvement.md` 4.1절)
    exact = models.SearchParams(exact=True)
    collection = config["index"]["collection"]
    with open_index(config["paths"]["index"]) as client:

        def dense_search(limit: int) -> list[Any]:
            return client.query_points(
                collection, query=vector, query_filter=acl, limit=limit, search_params=exact
            ).points

        if retrieval == "dense":
            return [chunk(p.payload, p.score) for p in dense_search(limit)]
        indices, values = embedding.bm25_query(question)
        candidates = cfg["top_k"] * HYBRID_CANDIDATES
        dense = dense_search(candidates)
        bm25 = client.query_points(
            collection,
            query=models.SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR,
            query_filter=acl,
            limit=candidates,
        ).points
    return rrf(dense, bm25, limit)


def chunk(payload: dict[str, Any], score: float) -> dict[str, Any]:
    """검색된 청크 하나: 색인의 payload에 검색 점수를 붙인다."""
    return {**payload, "score": score}


def rrf(dense: list[Any], bm25: list[Any], limit: int) -> list[dict[str, Any]]:
    """dense 후보와 BM25 후보를 RRF로 합쳐 상위 `limit`개 청크(payload와 RRF 점수)를 돌려준다.

    점수가 같으면 (1) 두 검색 중 더 좋은 순위, (2) dense 순위(dense에 없으면 가장 뒤), (3) 청크 ID
    순으로 줄을 세운다. Qdrant의 RRF는 동점 순서를 정하지 않아 같은 질문에도 결과가 바뀌었다 (#59).
    """
    ranks: dict[Any, dict[str, int]] = {}
    points: dict[Any, Any] = {}
    for name, found in (("dense", dense), ("bm25", bm25)):
        for rank, p in enumerate(found, 1):
            ranks.setdefault(p.id, {})[name] = rank
            points[p.id] = p
    # 분수로 더해 1/2 + 1/5와 1/3 + 1/3처럼 값이 같은 점수가 부동소수 오차로 갈리지 않게 한다
    scores = {i: sum(Fraction(1, RRF_CONSTANT + r) for r in rs.values()) for i, rs in ranks.items()}
    order = sorted(
        ranks,
        key=lambda i: (
            -scores[i],
            min(ranks[i].values()),
            ranks[i].get("dense", float("inf")),
            i,
        ),
    )
    return [chunk(points[i].payload, float(scores[i])) for i in order[:limit]]


def retrieval_of(config: dict[str, Any]) -> str:
    """config의 검색 방식(`ask.retrieval`). 없으면 `DEFAULT_RETRIEVAL`."""
    return config["ask"].get("retrieval", DEFAULT_RETRIEVAL)


def check_retrieval(retrieval: str) -> None:
    """검색 방식이 `RETRIEVALS`에 없으면 AskError."""
    if retrieval not in RETRIEVALS:
        raise AskError(
            f"ask.retrieval(검색 방식)은 {', '.join(RETRIEVALS)} 중 하나여야 한다: {retrieval}"
        )


def check_index(config: dict[str, Any]) -> None:
    """질의응답 전에 검색 방식·리랭커와 색인이 맞는지 확인한다. 모르는 검색 방식이나 리랭커이거나,
    hybrid인데 색인에 BM25 sparse 벡터가 없으면(#52 이전 형식) AskError."""
    retrieval = retrieval_of(config)
    check_retrieval(retrieval)
    check_reranker(reranker_of(config))
    if retrieval != "hybrid":
        return
    collection = config["index"]["collection"]
    with open_index(config["paths"]["index"]) as client:
        sparse = client.get_collection(collection).config.params.sparse_vectors or {}
    if SPARSE_VECTOR not in sparse:
        raise AskError(
            f"색인 {collection}에 BM25 sparse 벡터가 없어 hybrid 검색을 할 수 없다 "
            "(index를 다시 실행한다)"
        )


def reranker_of(config: dict[str, Any]) -> str:
    """config의 리랭커(`ask.reranker`). 없으면 `none`."""
    return config["ask"].get("reranker", "none")


def check_reranker(reranker: str) -> None:
    """리랭커가 `RERANKERS`에 없으면 AskError."""
    if reranker not in RERANKERS:
        raise AskError(
            f"ask.reranker(리랭커)는 {', '.join(RERANKERS)} 중 하나여야 한다: {reranker}"
        )


def rerank(
    config: dict[str, Any], question: str, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """검색 후보를 리랭커 점수(0~1)로 다시 매겨 상위 k개를 고른다.

    기준 점수(`ask.rerank_min_score`, 없으면 0.5)보다 낮은 청크는 뺀다. 점수가 같으면 원래 검색
    순위를 따른다. `none`은 그대로 통과한다.
    """
    reranker = reranker_of(config)
    check_reranker(reranker)
    if reranker == "none" or not chunks:
        return chunks
    scores = embedding.load_reranker(reranker).predict([(question, c["text"]) for c in chunks])
    min_score = config["ask"].get("rerank_min_score", DEFAULT_RERANK_MIN_SCORE)
    ranked = sorted(
        ({**c, "rerank_score": float(s)} for c, s in zip(chunks, scores, strict=True)),
        key=lambda c: -c["rerank_score"],
    )  # sorted는 안정 정렬이라 점수가 같으면 검색 순위가 유지된다
    return [c for c in ranked if c["rerank_score"] >= min_score][: config["ask"]["top_k"]]


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
