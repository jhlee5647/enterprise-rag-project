"""색인기: 문서 표의 본문을 PII 마스킹하고 청크로 잘라 임베딩해 Qdrant(로컬 모드)에 저장한다.

마스킹은 작업 프로세스 여러 개로 나눠 하고, 결과를 마스킹 결과 파일에 저장해 다음 색인에서 다시
쓴다. 다시 쓰는 조건은 문서 표 파일의 SHA-256, PII 종류 목록, spaCy 모델 이름, 이름 찾기 모델,
마스커 버전이 모두 같을 때다 (`docs/design/issue8-followup.md` 11절).
"""

import hashlib
import json
import logging
import multiprocessing
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from enterprise_rag import embedding, mask
from enterprise_rag.config import is_url

log = logging.getLogger(__name__)

# 이만큼씩 임베딩하고, 그 벡터를 `UPSERT_BATCH`개씩 저장한다. 저장하는 동안 GPU가 덜 쉬게 크게
# 묶는다 (`docs/design/issue8-followup.md` 13절)
EMBED_BATCH = 8192
UPSERT_BATCH = 1024
# 마스킹 결과 파일의 메타데이터에 다시 쓰는 조건을 적는 키
CONDITIONS_KEY = b"masking_conditions"
# 작업 프로세스에 한 번에 넘기는 문서 수. 문서 길이가 고르지 않아 작게 나눈다
DOCS_PER_TASK = 8


class IndexingError(Exception):
    """색인을 시작할 수 없을 때. 메시지를 그대로 사용자에게 보여 준다."""


class IndexServerError(Exception):
    """색인 위치가 서버 주소인데 Qdrant 서버에 연결할 수 없을 때. 메시지를 그대로 보여 준다."""


def index(config: dict[str, Any]) -> None:
    """문서 표를 읽어 색인을 새로 만든다. 이전 색인은 지운다."""
    cfg = config["index"]
    workers = config["pii"]["workers"]
    if not isinstance(workers, int) or workers < 1:
        raise IndexingError(f"pii.workers(마스킹 작업 프로세스 수)는 1 이상이어야 한다: {workers}")
    # 작업 프로세스 안에서 실패하면 원인이 가려지므로, 띄우기 전에 여기서 알린다
    mask.check_types(config["pii"]["types"])
    try:
        mask.check_name_model(mask.name_model(config))
    except ValueError as e:
        raise IndexingError(str(e)) from None
    try:
        embedding.check_dtype(config["embedding"]["dtype"])
    except ValueError as e:
        raise IndexingError(str(e)) from None
    with open_index(config["paths"]["index"]):
        pass  # 서버가 떠 있지 않으면 오래 걸리는 마스킹·임베딩 전에 멈춘다
    embedder = embedding.load_embedder(config)
    documents = pq.read_table(config["paths"]["documents"]).to_pylist()
    masked_pairs = masked_documents(config, documents)

    payloads = []
    for doc, (title, body) in zip(documents, masked_pairs, strict=True):
        for number, text in enumerate(
            chunk(body, embedder, cfg["chunk_size"], cfg["chunk_overlap"])
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

    collection = config["index"]["collection"]
    with open_index(config["paths"]["index"]) as client:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        for start in range(0, len(payloads), EMBED_BATCH):
            batch = payloads[start : start + EMBED_BATCH]
            vectors = embedder.embed_documents([p["text"] for p in batch])
            points = [
                models.PointStruct(id=start + i, vector=v, payload=p)
                for i, (v, p) in enumerate(zip(vectors, batch, strict=True))
            ]
            for offset in range(0, len(points), UPSERT_BATCH):
                client.upsert(collection, points[offset : offset + UPSERT_BATCH])
    log.info("문서 %d개, 청크 %d개를 색인했다", len(documents), len(payloads))


def masked_documents(
    config: dict[str, Any], documents: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """문서마다 마스킹한 (제목, 본문)을 문서 표 순서대로 돌려준다.

    조건이 같은 마스킹 결과 파일이 있으면 그것을 쓰고, 없거나 조건이 다르면 다시 마스킹해 덮어쓴다.
    """
    path = config["paths"]["masked"]
    conditions = masking_conditions(config)
    doc_ids = [d["doc_id"] for d in documents]
    if path.exists():
        saved = pq.read_table(path)
        meta = saved.schema.metadata or {}
        same = json.loads(meta.get(CONDITIONS_KEY, b"null")) == conditions
        if same and saved["doc_id"].to_pylist() == doc_ids:
            log.info("저장된 마스킹 결과를 썼다: 문서 %d건 (%s)", saved.num_rows, path)
            return list(zip(saved["title"].to_pylist(), saved["text"].to_pylist(), strict=True))

    workers = config["pii"]["workers"]
    pairs = [(d["title"], d["text"]) for d in documents]
    model, types = config["pii"]["spacy_model"], config["pii"]["types"]
    name_model = mask.name_model(config)
    if workers == 1:
        _start_worker(model, types, name_model)
        masked_pairs = [_mask_pair(p) for p in pairs]
    else:
        # 부모 프로세스가 불러 둔 임베딩 모델(GPU)을 물려받지 않도록 새로 띄운다.
        # 작업 프로세스가 시작하다 실패하면 멈춰 기다리지 않고 BrokenProcessPool로 끝난다.
        with ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker,
            initargs=(model, types, name_model),
        ) as pool:
            masked_pairs = list(pool.map(_mask_pair, pairs, chunksize=DOCS_PER_TASK))
    table = pa.table(
        {
            "doc_id": doc_ids,
            "title": [t for t, _ in masked_pairs],
            "text": [b for _, b in masked_pairs],
        }
    )
    # 쓰다가 멈춰도 깨진 파일이 남지 않도록 옆에 쓴 뒤 이름을 바꾼다
    partial = path.with_name(path.name + ".partial")
    pq.write_table(table.replace_schema_metadata({CONDITIONS_KEY: json.dumps(conditions)}), partial)
    partial.replace(path)
    log.info(
        "문서 %d건을 마스킹했다 (작업 프로세스 %d개, %s에 저장)", len(masked_pairs), workers, path
    )
    return masked_pairs


def masking_conditions(config: dict[str, Any]) -> dict[str, Any]:
    """마스킹 결과를 다시 쓰는 조건 다섯 가지."""
    digest = hashlib.sha256()
    with open(config["paths"]["documents"], "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {
        "documents_sha256": digest.hexdigest(),
        "pii_types": config["pii"]["types"],
        "spacy_model": config["pii"]["spacy_model"],
        "name_model": mask.name_model(config),
        "masker_version": mask.VERSION,
    }


_masker: mask.Masker | None = None


def _start_worker(spacy_model: str, types: list[str], name_model: str) -> None:
    """작업 프로세스마다 마스커를 한 번 불러 둔다."""
    global _masker
    _masker = mask.Masker(spacy_model, types, name_model)


def _mask_pair(pair: tuple[str, str]) -> tuple[str, str]:
    """(제목, 본문)을 이 프로세스의 마스커로 가린다."""
    title, text = pair
    return _masker.mask(title), _masker.mask(text)


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
def open_index(location: str | Path) -> Iterator[QdrantClient]:
    """색인을 연다. `http://`, `https://`로 시작하면 Qdrant 서버에 붙고, 아니면 폴더를 로컬 모드로
    연다. 로컬 모드는 폴더를 잠그므로 쓰고 나면 바로 닫는다.

    서버에 연결할 수 없으면 서버 주소와 띄우는 방법을 담아 `IndexServerError`.
    """
    if is_url(location):
        # 서버 버전은 scripts/qdrant-server.sh가 클라이언트와 같은 마이너 버전으로 고정한다.
        # 버전 확인을 켜 두면 서버가 없을 때 안내 메시지 앞에 경고가 한 번 더 찍힌다
        client = QdrantClient(url=str(location), check_compatibility=False)
        try:
            client.get_collections()
        except (ResponseHandlingException, UnexpectedResponse) as e:
            client.close()
            raise IndexServerError(
                f"Qdrant 서버에 연결할 수 없다: {location} ({e})\n"
                f"Pod 안에서 서버를 띄운다: scripts/qdrant-server.sh start"
            ) from None
    else:
        client = QdrantClient(path=str(location))
    try:
        yield client
    finally:
        client.close()
