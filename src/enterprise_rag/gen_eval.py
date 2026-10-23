"""평가셋 생성기: 부서마다 그 부서 문서의 마스킹된 청크로 RAGAS 질문을 만들어 검수용 CSV로 쓴다.

부서마다 따로 만들어 한 질문의 정답 문서가 모두 한 부서 안에 있게 한다(설계 기록 21번).
함께 마스킹 전 원문 청크에 NER·정규식이 찾은 PII를 적은 PII 정답셋 초안 CSV도 쓴다(#7).
"""

import csv
import logging
import random
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from enterprise_rag import embedding, mask
from enterprise_rag.departments import departments
from enterprise_rag.evaluate import (
    DOC_ID_SEPARATOR,
    EVAL_SET_COLUMNS,
    PII_ANSWER_COLUMNS,
    format_pii,
)
from enterprise_rag.index import COLLECTION, chunk, department_filter, open_index

log = logging.getLogger(__name__)


class GenEvalError(Exception):
    """평가셋 생성을 시작할 수 없을 때. 메시지를 그대로 사용자에게 보여 준다."""


# RAGAS 질문 생성기 이름 → 평가셋 CSV의 질문 유형
QUESTION_TYPES = {
    "single_hop_specific_query_synthesizer": "single_hop",
    "multi_hop_specific_query_synthesizer": "multi_hop_specific",
    "multi_hop_abstract_query_synthesizer": "multi_hop_abstract",
}
# RAGAS는 다중 홉 질문의 근거 청크 앞에 `<1-hop>` 같은 머리말과 빈 줄을 붙인다
HOP_PREFIX = re.compile(r"\A<\d+-hop>\n\n")


def gen_eval(config: dict[str, Any], pii_only: bool = False) -> Path:
    """부서마다 질문을 만들어 config의 평가셋 경로에 쓰고 그 경로를 돌려준다. `keep`은 비워 둔다.

    PII 정답셋 초안도 config의 PII 정답셋 경로에 쓴다. 두 파일 중 하나라도 이미 있으면 사람이
    고친 내용을 지우지 않도록 생성하기 전에 멈춘다.

    `pii_only`이면 평가셋은 만들지도 건드리지도 않고 PII 정답셋만 만들거나 늘려 그 경로를 돌려준다.
    """
    pii_path = config["paths"]["pii_answers"]
    if pii_only:
        write_pii_draft(config)
        return pii_path
    path = config["paths"]["eval_set"]
    if path.exists():
        raise GenEvalError(f"평가셋 파일이 이미 있다: {path} (지우거나 옮긴 뒤 다시 실행한다)")
    if pii_path.exists():
        raise GenEvalError(
            f"PII 정답셋 파일이 이미 있다: {pii_path} (지우거나 옮긴 뒤 다시 실행한다)"
        )
    cfg = config["gen_eval"]
    rows = []
    for department in departments(config["collect"]["departments"]):
        chunks = pick_documents(
            department_chunks(config, department), cfg["docs_per_department"], cfg["seed"]
        )
        doc_id_by_text = {c["text"]: c["doc_id"] for c in chunks}
        samples = synthesize(config, list(doc_id_by_text), cfg["questions_per_department"])
        made = [question_row(s, department, doc_id_by_text) for s in samples]
        kept = [r for r in made if r is not None]
        if len(kept) < len(made):
            log.warning(
                "문서 ID를 되찾지 못한 질문 %d개를 뺐다 (%s)", len(made) - len(kept), department
            )
        log.info(
            "%s: 문서 %d건으로 질문 %d개를 만들었다",
            department,
            len(set(doc_id_by_text.values())),
            len(kept),
        )
        rows += kept

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_SET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("질문 %d개를 %s에 썼다", len(rows), path)
    write_pii_draft(config)
    return path


def write_pii_draft(config: dict[str, Any]) -> None:
    """문서 표의 원문을 청크로 잘라 시드로 청크 N개를 고르고, 찾은 PII를 적어 초안 CSV로 쓴다.

    PII는 config의 PII 종류 목록과 상관없이 모든 종류를 찾는다. 정답셋은 마스킹 설정을 바꿔도
    그대로 쓰는 자료이기 때문이다.

    CSV가 이미 있으면 그 행은 사람이 확정한 행이므로 원문과 PII 목록을 그대로 두고 새 청크 행만
    더한다. 같은 시드로 뽑으면 앞의 청크가 기존 청크와 같다(`random.sample`은 뽑는 수를 늘려도
    앞쪽 결과가 같다). 행마다 뽑힌 순서로 개발용(`dev`)과 측정용(`test`)을 번갈아 붙인다.
    그래서 처음부터 만들든 늘리든 같은 청크에는 같은 표시가 붙고, 홀수면 개발용이 하나 많다.
    """
    cfg = config["gen_eval"]
    path = config["paths"]["pii_answers"]
    confirmed = read_pii_rows(path) if path.exists() else {}
    if cfg["pii_chunks"] < len(confirmed):
        raise GenEvalError(
            f"PII 정답셋에 이미 {len(confirmed)}행이 있는데 청크 수를 {cfg['pii_chunks']}개로 "
            f"줄였다: {path} (확정된 행을 지우려면 CSV에서 직접 지운다)"
        )
    chunks = source_chunks(config)
    text_of = {(c["doc_id"], c["chunk"]): c["text"] for c in chunks}
    changed = [
        doc_id for (doc_id, n), r in confirmed.items() if text_of.get((doc_id, n)) != r["text"]
    ]
    if changed:
        raise GenEvalError(
            f"PII 정답셋의 원문이 지금 청크와 다른 행이 있다: 문서 {', '.join(changed)} "
            "(청크 설정이 바뀌었는지 확인한다)"
        )
    picked = random.Random(cfg["seed"]).sample(chunks, min(cfg["pii_chunks"], len(chunks)))
    keys = {(c["doc_id"], c["chunk"]) for c in picked}
    unpicked = [doc_id for doc_id, n in confirmed if (doc_id, n) not in keys]
    if unpicked:
        raise GenEvalError(
            f"PII 정답셋의 행이 시드로 뽑은 청크에 없다: 문서 {', '.join(unpicked)} "
            "(seed가 바뀌었는지 확인한다)"
        )

    new = [c for c in picked if (c["doc_id"], c["chunk"]) not in confirmed]
    if new:  # 새 행이 없으면 spaCy 모델을 불러오지 않는다
        finder = mask.Masker(config["pii"]["spacy_model"], list(mask.TYPES))
        pii_of = {(c["doc_id"], c["chunk"]): format_pii(finder.find(c["text"])) for c in new}
    rows = [
        {
            **c,
            "pii": confirmed[key]["pii"] if key in confirmed else pii_of[key],
            "split": "dev" if order % 2 == 0 else "test",
        }
        for order, c in enumerate(picked)
        for key in [(c["doc_id"], c["chunk"])]
    ]
    rows.sort(key=lambda r: (r["doc_id"], r["chunk"]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PII_ANSWER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info(
        "원문 청크 %d개 중 %d개로 PII 정답셋 초안을 %s에 썼다 (확정된 행 %d개, 새 행 %d개)",
        len(chunks),
        len(rows),
        path,
        len(confirmed),
        len(new),
    )


def read_pii_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """PII 정답셋 CSV의 행을 (문서 ID, 청크 번호)로 찾을 수 있게 읽는다."""
    with open(path, newline="") as f:
        return {(r["doc_id"], int(r["chunk"])): r for r in csv.DictReader(f)}


def source_chunks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """문서 표의 마스킹 전 원문을 `index`와 같은 설정으로 잘라 청크 목록으로 돌려준다."""
    embedder = embedding.load_embedder(config)
    return [
        {"doc_id": doc["doc_id"], "chunk": number, "text": text}
        for doc in pq.read_table(
            config["paths"]["documents"], columns=["doc_id", "text"]
        ).to_pylist()
        for number, text in enumerate(
            chunk(
                doc["text"],
                embedder,
                config["index"]["chunk_size"],
                config["index"]["chunk_overlap"],
            )
        )
    ]


def question_row(
    sample: dict[str, Any], department: str, doc_id_by_text: dict[str, str]
) -> dict[str, str] | None:
    """RAGAS 결과 한 행을 평가셋 CSV 한 행으로 바꾼다. 정답 문서 ID는 근거 청크 내용으로 되찾는다.

    근거 중 하나라도 어느 청크와도 같지 않으면 정답 문서를 알 수 없으므로 None을 돌려준다.
    """
    doc_ids = [doc_id_by_text.get(HOP_PREFIX.sub("", c)) for c in sample["reference_contexts"]]
    if not doc_ids or None in doc_ids:
        return None
    return {
        "question": sample["user_input"],
        "answer": sample["reference"],
        "doc_ids": DOC_ID_SEPARATOR.join(dict.fromkeys(doc_ids)),
        "department": department,
        "question_type": QUESTION_TYPES[sample["synthesizer_name"]],
        "keep": "",
    }


def pick_documents(chunks: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """문서 `n`건을 시드로 무작위로 골라 그 문서들의 청크만 남긴다. 순서는 그대로 둔다."""
    doc_ids = sorted({c["doc_id"] for c in chunks})
    picked = set(random.Random(seed).sample(doc_ids, min(n, len(doc_ids))))
    return [c for c in chunks if c["doc_id"] in picked]


def department_chunks(config: dict[str, Any], department: str) -> list[dict[str, Any]]:
    """색인에서 그 부서가 허용 부서에 들어 있는 문서의 청크를 문서 ID·청크 순서대로 읽는다."""
    with open_index(config["paths"]["index"]) as client:
        chunks, offset = [], None
        while True:
            points, offset = client.scroll(
                COLLECTION,
                scroll_filter=department_filter(department),
                limit=1000,
                offset=offset,
                with_vectors=False,
            )
            chunks += [p.payload for p in points]
            if offset is None:
                break
    return sorted(chunks, key=lambda c: (c["doc_id"], c["chunk"]))


def synthesize(config: dict[str, Any], chunks: list[str], size: int) -> list[dict[str, Any]]:
    """RAGAS로 청크에서 질문 최대 `size`개를 만들어 `Testset.to_list()`의 행 모양으로 돌려준다.

    LLM은 `ask`와 같은 localhost vLLM, 임베딩도 `ask`와 같은 모델이다(ADR-0002).
    작은 로컬 모델은 RAGAS가 요구하는 JSON 형식을 자주 틀리므로, 질문 하나가 실패하면 그 질문만
    버리고, 한 유형 전체가 실패하면(예: 엮을 청크가 없는 다중 홉) 그 유형만 건너뛴다.
    테스트는 이 함수를 RAGAS 출력과 같은 모양의 가짜로 바꾼다.
    """
    # RAGAS는 무겁고 LangChain 경고를 많이 내서 쓸 때만 불러온다
    import httpx
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )
    from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config["llm"]["model"],
            base_url=config["llm"]["base_url"],
            api_key="EMPTY",  # vLLM은 키를 보지 않는다
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            # RAGAS는 단계마다 새 이벤트 루프를 만든다. 이전 루프에서 연 연결을 다시 쓰면
            # Connection error가 나므로 연결을 재사용하지 않는다(localhost라 비용이 작다).
            http_async_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=0), timeout=300
            ),
        )
    )
    embeddings = LangchainEmbeddingsWrapper(_LangchainEmbeddings(embedding.load_embedder(config)))
    kg = KnowledgeGraph(
        nodes=[
            Node(type=NodeType.CHUNK, properties={"page_content": c, "document_metadata": {}})
            for c in chunks
        ]
    )
    apply_transforms(kg, default_transforms_for_prechunked(llm=llm, embedding_model=embeddings))
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)

    synthesizers = {
        "single_hop": SingleHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_specific": MultiHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_abstract": MultiHopAbstractQuerySynthesizer(llm=llm),
    }
    samples = []
    for name, n in split(size, config["gen_eval"]["query_distribution"]).items():
        synthesizer = synthesizers[name]
        try:
            # 유형마다 따로 실행한다. RAGAS generate()는 질문 하나만 실패해도 전체가 멈추므로
            # 실행기를 받아 직접 돌리고 실패한 질문(NaN)을 거른다.
            executor = generator.generate(
                testset_size=n,
                query_distribution=[(synthesizer, 1.0)],
                raise_exceptions=False,
                return_executor=True,
            )
        except Exception as e:
            log.warning("%s 질문을 만들지 못해 이 유형은 건너뛴다: %r", name, e)
            continue
        made = [s for s in executor.results() if not isinstance(s, float)]
        log.info("%s 질문 %d개 중 %d개를 만들었다", name, n, len(made))
        samples += [
            {**s.model_dump(exclude_none=True), "synthesizer_name": synthesizer.name} for s in made
        ]
    return samples


def split(size: int, shares: dict[str, float]) -> dict[str, int]:
    """질문 `size`개를 유형 비율대로 나눈다. 합이 `size`가 되게 나머지는 소수부가 큰 유형부터 준다.

    예: 15개를 50/25/25로 나누면 7.5, 3.75, 3.75 → 7, 4, 4.
    """
    exact = {name: size * share for name, share in shares.items()}
    counts = {name: int(x) for name, x in exact.items()}
    by_remainder = sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in by_remainder[: size - sum(counts.values())]:
        counts[name] += 1
    return counts


class _LangchainEmbeddings:
    """`embedding.Embedder`를 RAGAS가 받는 LangChain `Embeddings` 모양으로 감싼다."""

    def __init__(self, embedder: embedding.Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)
