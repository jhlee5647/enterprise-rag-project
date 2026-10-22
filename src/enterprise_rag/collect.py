"""표본 수집기: IDL Solr에서 표본을 뽑고 S3에서 본문을 받아 문서 표로 저장한다."""

import logging
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def collect(config: dict[str, Any]) -> None:
    cfg = config["collect"]
    solr_url = config["idl"]["solr_url"]
    s3_url = config["idl"]["s3_url"].rstrip("/")
    rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=60) as client:
        for topic, department in cfg["departments"].items():
            for doc_type in cfg["doc_types"]:
                q = f'collectioncode:juul AND type:{doc_type} AND topic:"{topic}"'
                for doc in _candidates(client, solr_url, q, cfg["seed"]):
                    text = client.get(_body_url(s3_url, doc["id"], "ocr")).text
                    rows.append(
                        {
                            "doc_id": doc["id"],
                            "departments": [department],
                            "text": text,
                        }
                    )

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


def _body_url(s3_url: str, doc_id: str, ext: str) -> str:
    prefix = "/".join(doc_id[:4])
    return f"{s3_url}/{prefix}/{doc_id}/{doc_id}.{ext}"


def _write(paths: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    paths["documents"].parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), paths["documents"])
    paths["sample_ids"].write_text("".join(r["doc_id"] + "\n" for r in rows))
