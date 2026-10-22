"""표본 수집기: IDL Solr에서 표본을 뽑고 S3에서 본문을 받아 문서 표로 저장한다."""

import logging
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from enterprise_rag.departments import allowed_departments

log = logging.getLogger(__name__)

# 문서 표: 수집과 색인 사이의 공식 계약. 한 행이 문서 하나다.
# 보관자·작성자·수신자 같은 인명 메타데이터는 넣지 않는다.
DOCUMENT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("bates", pa.string()),
        ("title", pa.string()),
        ("doc_type", pa.list_(pa.string())),
        ("topic", pa.string()),  # IDL 토픽, 받은 값 그대로
        ("departments", pa.list_(pa.string())),  # 허용 부서
        ("text", pa.string()),  # 본문, PII 마스킹 전
    ]
)


def collect(config: dict[str, Any]) -> None:
    cfg = config["collect"]
    solr_url = config["idl"]["solr_url"]
    s3_url = config["idl"]["s3_url"].rstrip("/")
    rows: list[dict[str, Any]] = []
    visited: set[str] = set()  # 이미 판정한 문서 ID. 뽑혔든 버려졌든 다른 칸에서 다시 보지 않는다.

    with httpx.Client(timeout=60) as client:
        for topic, department in cfg["departments"].items():
            for doc_type in cfg["doc_types"]:
                q = _cell_query(topic, doc_type, cfg["excluded_types"])
                collected = 0
                for doc in _candidates(client, solr_url, q, cfg["seed"]):
                    if doc["id"] in visited:
                        continue
                    visited.add(doc["id"])
                    departments = allowed_departments(doc.get("topic", ""), cfg["departments"])
                    if not departments:
                        continue
                    text = _fetch_body(client, s3_url, doc["id"])
                    if text is None:
                        log.warning("%s: 본문 파일(.ocr, .txt)이 없어 건너뛴다", doc["id"])
                        continue
                    rows.append(
                        {
                            "doc_id": doc["id"],
                            "bates": doc.get("bates", ""),
                            "title": doc.get("title", ""),
                            "doc_type": doc["type"],
                            "topic": doc["topic"],
                            "departments": departments,
                            "text": text,
                        }
                    )
                    collected += 1
                    if collected == cfg["quota"]:
                        break
                quota, shortfall = cfg["quota"], cfg["quota"] - collected
                if shortfall:
                    log.warning(
                        "%s/%s: 목표 %d, 수집 %d, 부족 %d",
                        department,
                        doc_type,
                        quota,
                        collected,
                        shortfall,
                    )
                else:
                    log.info("%s/%s: 목표 %d, 수집 %d", department, doc_type, quota, collected)

    _write(config["paths"], rows)


def _cell_query(topic: str, doc_type: str, excluded_types: list[str]) -> str:
    """칸 하나의 Solr 검색 조건. IDL Solr는 `fq`를 제대로 적용하지 않아 모든 조건을 `q`에 넣는다."""
    excluded = " OR ".join(excluded_types)
    return f'collectioncode:juul AND type:{doc_type} AND topic:"{topic}" AND NOT type:({excluded})'


def _candidates(client: httpx.Client, solr_url: str, q: str, seed: int):
    start = 0
    while True:
        params = {"q": q, "sort": f"random_{seed} asc", "start": start, "wt": "json"}
        response = client.get(solr_url, params=params)
        response.raise_for_status()
        body = response.json()["response"]
        docs = body["docs"]
        yield from docs
        start += len(docs)
        if not docs or start >= body["numFound"]:
            return


def _fetch_body(client: httpx.Client, s3_url: str, doc_id: str) -> str | None:
    """`.ocr` 본문을 받고, 없으면 `.txt`를 받는다. 둘 다 없으면 None."""
    for ext in ("ocr", "txt"):
        response = client.get(_body_url(s3_url, doc_id, ext))
        if response.status_code == 404:
            continue
        response.raise_for_status()
        return response.text
    return None


def _body_url(s3_url: str, doc_id: str, ext: str) -> str:
    prefix = "/".join(doc_id[:4])
    return f"{s3_url}/{prefix}/{doc_id}/{doc_id}.{ext}"


def _write(paths: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    paths["documents"].parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=DOCUMENT_SCHEMA), paths["documents"])
    paths["sample_ids"].write_text("".join(r["doc_id"] + "\n" for r in rows))
