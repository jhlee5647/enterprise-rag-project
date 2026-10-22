"""표본 수집기: IDL Solr에서 표본을 뽑고 S3에서 본문을 받아 문서 표로 저장한다."""

import logging
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from enterprise_rag.departments import allowed_departments

log = logging.getLogger(__name__)


def collect(config: dict[str, Any]) -> None:
    cfg = config["collect"]
    solr_url = config["idl"]["solr_url"]
    s3_url = config["idl"]["s3_url"].rstrip("/")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    with httpx.Client(timeout=60) as client:
        for topic in cfg["departments"]:
            for doc_type in cfg["doc_types"]:
                q = f'collectioncode:juul AND type:{doc_type} AND topic:"{topic}"'
                collected = 0
                for doc in _candidates(client, solr_url, q, cfg["seed"]):
                    if doc["id"] in seen:
                        continue
                    departments = allowed_departments(doc.get("topic", ""), cfg["departments"])
                    if not departments:
                        continue
                    text = _fetch_body(client, s3_url, doc["id"])
                    if text is None:
                        log.warning("%s: 본문 파일(.ocr, .txt)이 없어 건너뛴다", doc["id"])
                        continue
                    rows.append({"doc_id": doc["id"], "departments": departments, "text": text})
                    seen.add(doc["id"])
                    collected += 1
                    if collected == cfg["quota"]:
                        break

    _write(config["paths"], rows)


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
    pq.write_table(pa.Table.from_pylist(rows), paths["documents"])
    paths["sample_ids"].write_text("".join(r["doc_id"] + "\n" for r in rows))
