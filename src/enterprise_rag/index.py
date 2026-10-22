"""색인기: 문서 표의 본문을 PII 마스킹하고 청크로 잘라 임베딩해 Qdrant(로컬 모드)에 저장한다."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from enterprise_rag import embedding, mask

log = logging.getLogger(__name__)

COLLECTION = "chunks"
BATCH_SIZE = 64


def index(config: dict[str, Any]) -> None:
    """문서 표를 읽어 색인을 새로 만든다. 이전 색인은 지운다."""
    cfg = config["index"]
    embedder = embedding.load_embedder(config)
    masker = mask.load_masker(config)
    documents = pq.read_table(config["paths"]["documents"]).to_pylist()

    payloads = []
    for doc in documents:
        title = masker.mask(doc["title"])
        for number, text in enumerate(
            chunk(masker.mask(doc["text"]), embedder, cfg["chunk_size"], cfg["chunk_overlap"])
        ):
            payload = {
                "departments": doc["departments"],
                "doc_id": doc["doc_id"],
                "bates": doc["bates"],
                "title": title,
                "chunk": number,
                "text": text,
            }
            payloads.append(payload)

    with open_index(config["paths"]["index"]) as client:
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        for start in range(0, len(payloads), BATCH_SIZE):
            batch = payloads[start : start + BATCH_SIZE]
            vectors = embedder.embed_documents([p["text"] for p in batch])
            client.upsert(
                COLLECTION,
                [
                    models.PointStruct(id=start + i, vector=v, payload=p)
                    for i, (v, p) in enumerate(zip(vectors, batch, strict=True))
                ],
            )
    log.info("문서 %d개, 청크 %d개를 색인했다", len(documents), len(payloads))


def chunk(text: str, embedder: embedding.Embedder, size: int, overlap: int) -> list[str]:
    """본문을 `size`토큰씩 자르고 앞 청크와 `overlap`토큰을 겹친다. 내용은 정리하지 않는다."""
    if not 0 <= overlap < size:
        raise ValueError(f"청크 겹침({overlap})은 0 이상, 청크 길이({size}) 미만이어야 한다")
    spans = embedder.token_spans(text)
    chunks = []
    start = 0
    while start < len(spans):
        end = min(start + size, len(spans))
        chunks.append(text[spans[start][0] : spans[end - 1][1]])
        if end == len(spans):
            break
        start = end - overlap
    return chunks


def department_filter(department: str) -> models.Filter:
    """허용 부서에 `department`가 들어 있는 청크만 남기는 Qdrant 필터."""
    return models.Filter(
        must=[models.FieldCondition(key="departments", match=models.MatchValue(value=department))]
    )


@contextmanager
def open_index(path: Path) -> Iterator[QdrantClient]:
    """색인 폴더를 연다. 로컬 모드는 폴더를 잠그므로 쓰고 나면 바로 닫는다."""
    client = QdrantClient(path=str(path))
    try:
        yield client
    finally:
        client.close()
