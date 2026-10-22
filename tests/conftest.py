import json

import pyarrow.parquet as pq
import pytest
import yaml

from enterprise_rag import embedding
from enterprise_rag.cli import main
from tests.fake_idl import FakeIDL
from tests.fake_models import FakeEmbedder, FakeVLLM


def make_config(out_dir, quota=2, seed=42):
    return {
        "paths": {
            "documents": str(out_dir / "documents.parquet"),
            "sample_ids": str(out_dir / "sample_ids.txt"),
            "collect_log": str(out_dir / "collect.log"),
            "index": str(out_dir / "index"),
        },
        "collect": {
            "seed": seed,
            "quota": quota,
            "doc_types": ["email", "document", "presentation"],
            "excluded_types": [
                "publication",
                "article",
                "news",
                "transcript",
                "video",
                "audio",
            ],
            "departments": {
                "Marketing": "marketing",
                "Government & Public Affairs": "regulatory",
                "Product Design & Quality": "product",
            },
        },
        "index": {"chunk_size": 512, "chunk_overlap": 64},
        "embedding": {"model": "fake-embedding"},
        "ask": {"acl_filter": True, "retrieval": "dense", "reranker": "none", "top_k": 5},
        "llm": {"model": "fake-llm", "base_url": "http://127.0.0.1:9/v1"},
    }


def write_config(path, config):
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


class Collected:
    """`collect` 실행 결과: 문서 표의 행과 문서 ID 목록."""

    def __init__(self, documents_path, sample_ids_path, collect_log_path):
        self.table = pq.read_table(documents_path)
        self.rows = self.table.to_pylist()
        self.by_id = {r["doc_id"]: r for r in self.rows}
        self.sample_ids = sample_ids_path.read_text().split()
        self.log_lines = collect_log_path.read_text().splitlines()


@pytest.fixture
def run_collect(tmp_path):
    """가짜 IDL 서버를 띄우고 config를 써서 `collect`를 실행한다."""

    def run(cells, quota=2, seed=42, out="out"):
        out_dir = tmp_path / out
        config = make_config(out_dir, quota=quota, seed=seed)
        with FakeIDL(cells) as idl:
            config["idl"] = {"solr_url": idl.solr_url, "s3_url": idl.s3_url}
            config_path = tmp_path / f"{out}.yaml"
            write_config(config_path, config)
            assert main(["--config", str(config_path), "collect"]) == 0
        result = Collected(
            out_dir / "documents.parquet", out_dir / "sample_ids.txt", out_dir / "collect.log"
        )
        result.solr_requests = idl.solr_requests
        return result

    return run


# 질의응답 fixture: 문서 A(inca0001), B(pmtb0001), C(agec0001)와 기존 문서 몇 개
PIPELINE_CELLS = {
    ("Marketing", "email"): ["mkea0002", "mkea0003"],
    ("Marketing", "document"): ["inca0001", "agec0001"],
    ("Government & Public Affairs", "email"): ["shrm0001"],
    ("Government & Public Affairs", "document"): ["pmtb0001", "txtd0001"],
    ("Product Design & Quality", "presentation"): ["prdp0001"],
}


class Asked:
    """`ask` 실행 결과: 종료 코드, 표준 출력(JSON이면 풀어 둔다), 표준 오류."""

    def __init__(self, code, out, err):
        self.code, self.err = code, err
        self.output = json.loads(out) if code == 0 else None

    @property
    def answer(self):
        return self.output["answer"]

    @property
    def citations(self):
        return self.output["citations"]

    @property
    def cited_ids(self):
        return [c["doc_id"] for c in self.citations]

    @property
    def retrieved_ids(self):
        return [c["doc_id"] for c in self.output["retrieved"]]


@pytest.fixture
def pipeline(run_collect, tmp_path, monkeypatch, capsys):
    """fixture로 `collect` → `index`를 실행하고, `ask`를 부르는 함수를 돌려준다.

    임베딩은 가짜로 바꾸고, LLM은 가짜 vLLM 서버로 보낸다.
    """
    monkeypatch.setattr(embedding, "load_embedder", lambda config: FakeEmbedder())
    run_collect(PIPELINE_CELLS, quota=2, out="out")
    config_path = tmp_path / "out.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert main(["--config", str(config_path), "index"]) == 0

    with FakeVLLM() as vllm:
        config["llm"]["base_url"] = vllm.base_url

        def ask(dept, question, acl_filter=True):
            config["ask"]["acl_filter"] = acl_filter
            write_config(config_path, config)
            capsys.readouterr()
            code = main(["--config", str(config_path), "ask", "--dept", dept, question])
            captured = capsys.readouterr()
            return Asked(code, captured.out, captured.err)

        yield ask
