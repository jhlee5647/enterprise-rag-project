"""평가셋 생성기: 부서마다 그 부서 문서의 마스킹된 청크로 RAGAS 질문을 만들어 검수용 CSV로 쓴다.

부서마다 따로 만들어 한 질문의 정답 문서가 모두 한 부서 안에 있게 한다(설계 기록 21번).
"""

import csv
import logging
import random
import re
from pathlib import Path
from typing import Any

from qdrant_client import models

from enterprise_rag import embedding
from enterprise_rag.departments import departments
from enterprise_rag.evaluate import DOC_ID_SEPARATOR, EVAL_SET_COLUMNS
from enterprise_rag.index import COLLECTION, open_index

log = logging.getLogger(__name__)

# RAGAS 질문 생성기 이름 → 평가셋 CSV의 질문 유형
QUESTION_TYPES = {
    "single_hop_specific_query_synthesizer": "single_hop",
    "multi_hop_specific_query_synthesizer": "multi_hop_specific",
    "multi_hop_abstract_query_synthesizer": "multi_hop_abstract",
}
# RAGAS는 다중 홉 질문의 근거 청크 앞에 `<1-hop>` 같은 머리말과 빈 줄을 붙인다
HOP_PREFIX = re.compile(r"\A<\d+-hop>\n\n")


def gen_eval(config: dict[str, Any]) -> Path:
    """부서마다 질문을 만들어 config의 평가셋 경로에 쓰고 그 경로를 돌려준다. `keep`은 비워 둔다."""
    cfg = config["gen_eval"]
    rows = []
    for department in departments(config["collect"]["departments"]):
        chunks = pick_documents(
            department_chunks(config, department), cfg["docs_per_department"], cfg["seed"]
        )
        doc_of = {c["text"]: c["doc_id"] for c in chunks}
        samples = synthesize(config, list(doc_of), cfg["questions_per_department"])
        made = [question_row(s, department, doc_of) for s in samples]
        kept = [r for r in made if r is not None]
        if len(kept) < len(made):
            log.warning(
                "문서 ID를 되찾지 못한 질문 %d개를 뺐다 (%s)", len(made) - len(kept), department
            )
        log.info(
            "%s: 문서 %d건으로 질문 %d개를 만들었다",
            department,
            len(set(doc_of.values())),
            len(kept),
        )
        rows += kept

    path = config["paths"]["eval_set"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_SET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("질문 %d개를 %s에 썼다", len(rows), path)
    return path


def question_row(
    sample: dict[str, Any], department: str, doc_of: dict[str, str]
) -> dict[str, str] | None:
    """RAGAS 결과 한 행을 평가셋 CSV 한 행으로 바꾼다. 정답 문서 ID는 근거 청크 내용으로 되찾는다.

    근거 중 하나라도 어느 청크와도 같지 않으면 정답 문서를 알 수 없으므로 None을 돌려준다.
    """
    doc_ids = [doc_of.get(HOP_PREFIX.sub("", c)) for c in sample["reference_contexts"]]
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
    only = models.Filter(
        must=[models.FieldCondition(key="departments", match=models.MatchValue(value=department))]
    )
    with open_index(config["paths"]["index"]) as client:
        chunks, offset = [], None
        while True:
            points, offset = client.scroll(
                COLLECTION, scroll_filter=only, limit=1000, offset=offset, with_vectors=False
            )
            chunks += [p.payload for p in points]
            if offset is None:
                break
    return sorted(chunks, key=lambda c: (c["doc_id"], c["chunk"]))


def synthesize(config: dict[str, Any], chunks: list[str], size: int) -> list[dict[str, Any]]:
    """RAGAS로 청크에서 질문 최대 `size`개를 만들어 `Testset.to_list()`의 행으로 돌려준다.

    LLM은 `ask`와 같은 localhost vLLM, 임베딩도 `ask`와 같은 모델이다(ADR-0002).
    질문 하나를 만들다 LLM이 형식을 틀리면 그 질문만 버리고 나머지는 계속 만든다.
    테스트는 이 함수를 RAGAS 출력과 같은 모양의 가짜로 바꾼다.
    """
    # RAGAS는 무겁고 LangChain 경고를 많이 내서 쓸 때만 불러온다
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
        )
    )
    embeddings = LangchainEmbeddingsWrapper(_LangchainEmbeddings(embedding.load_embedder(config)))

    # 지식 그래프를 먼저 만들어, 엮을 청크가 없는 질문 유형(예: 다중 홉)은 비율에서 뺄 수 있게 한다
    kg = KnowledgeGraph(
        nodes=[
            Node(type=NodeType.CHUNK, properties={"page_content": c, "document_metadata": {}})
            for c in chunks
        ]
    )
    apply_transforms(kg, default_transforms_for_prechunked(llm=llm, embedding_model=embeddings))

    synthesizers = {
        "single_hop": SingleHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_specific": MultiHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_abstract": MultiHopAbstractQuerySynthesizer(llm=llm),
    }
    distribution = []
    for name, share in config["gen_eval"]["query_distribution"].items():
        if synthesizers[name].get_node_clusters(kg):
            distribution.append((synthesizers[name], share))
        else:
            log.warning("%s 질문을 만들 청크 묶음이 없어 이 유형은 빼고 만든다", name)
    total = sum(share for _, share in distribution)
    distribution = [(s, share / total) for s, share in distribution]

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)
    testset = generator.generate(
        testset_size=size, query_distribution=distribution, raise_exceptions=False
    )
    # 만들다 실패한 질문은 질문 칸이 비어(NaN) 나온다
    return [s for s in testset.to_list() if isinstance(s.get("user_input"), str)]


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
