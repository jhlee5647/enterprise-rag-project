import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from enterprise_rag import embedding
from enterprise_rag.cli import main
from enterprise_rag.evaluate import EVAL_SET_COLUMNS, PII_ANSWER_COLUMNS
from enterprise_rag.index import COLLECTION, open_index
from tests.fake_idl import FakeIDL
from tests.fake_models import FakeEmbedder, FakeVLLM


def make_config(out_dir, quota=2, seed=42, max_chars=1_000_000):
    return {
        "paths": {
            "documents": str(out_dir / "documents.parquet"),
            "sample_ids": str(out_dir / "sample_ids.txt"),
            "collect_log": str(out_dir / "collect.log"),
            "index": str(out_dir / "index"),
            "eval_set": str(out_dir / "eval_set.csv"),
            "runs": str(out_dir / "runs"),
            "pii_answers": str(out_dir / "pii_answers.csv"),
        },
        "collect": {
            "seed": seed,
            "quota": quota,
            "max_chars": max_chars,
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
        "pii": {"types": ["NAME", "EMAIL", "PHONE"], "spacy_model": "en_core_web_lg"},
        "embedding": {"model": "fake-embedding"},
        "ask": {"acl_filter": True, "retrieval": "dense", "reranker": "none", "top_k": 5},
        "llm": {"model": "fake-llm", "base_url": "http://127.0.0.1:9/v1"},
        "gen_eval": {
            "seed": 42,
            "docs_per_department": 30,
            "questions_per_department": 15,
            "pii_chunks": 50,
            "query_distribution": {
                "single_hop": 0.5,
                "multi_hop_specific": 0.25,
                "multi_hop_abstract": 0.25,
            },
        },
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

    def run(cells, quota=2, seed=42, out="out", max_chars=1_000_000):
        out_dir = tmp_path / out
        config = make_config(out_dir, quota=quota, seed=seed, max_chars=max_chars)
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


# 질의응답 fixture: 문서 A(inca0001), B(pmtb0001), C(agec0001), D(crep0001),
# PII 마스킹 문서 P(pnot0001)와 기존 문서 몇 개
PIPELINE_CELLS = {
    ("Marketing", "email"): ["mkea0002", "mkea0003"],
    ("Marketing", "document"): ["inca0001", "agec0001"],
    ("Marketing", "presentation"): ["pnot0001", "crep0001"],
    ("Government & Public Affairs", "email"): ["shrm0001", "jdoe0001"],
    ("Government & Public Affairs", "document"): ["pmtb0001", "txtd0001"],
    ("Product Design & Quality", "document"): ["fdar0001"],
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


class EvalRun:
    """`eval` 실행 결과 폴더: 지표 JSON, 실행별 기록 파일, config 사본."""

    def __init__(self, folder):
        self.folder = folder
        self.metrics = json.loads((folder / "metrics.json").read_text())
        self.records = [json.loads(line) for line in (folder / "records.jsonl").open()]
        self.config_copy = folder / "config.yaml"

    def record(self, question, department):
        [found] = [
            r for r in self.records if r["question"] == question and r["department"] == department
        ]
        return found


class Pipeline:
    """`collect`를 마친 config로 `index`와 `ask`를 실행한다. 부르면 `ask`를 실행한다."""

    def __init__(self, config_path, config, vllm, capsys):
        self.config_path, self.config, self.vllm, self.capsys = config_path, config, vllm, capsys

    def index(self, pii_types=None):
        if pii_types is not None:
            self.config["pii"]["types"] = pii_types
        write_config(self.config_path, self.config)
        assert main(["--config", str(self.config_path), "index"]) == 0

    def __call__(self, dept, question, acl_filter=True):
        self._write_config(acl_filter=acl_filter)
        self.capsys.readouterr()
        code = main(["--config", str(self.config_path), "ask", "--dept", dept, question])
        captured = self.capsys.readouterr()
        return Asked(code, captured.out, captured.err)

    def eval(self, rows, run_name, acl_filter=True, top_k=5, expect_code=0, pii_answers=None):
        """평가셋 CSV(와 `pii_answers`가 있으면 PII 정답셋 CSV)를 쓰고 `eval --run-name`을
        실행해 실행 결과 폴더를 돌려준다.

        `expect_code`가 0이 아니면 그 종료 코드로 끝나는지 확인하고 종료 코드를 돌려준다.
        """
        self._write_config(acl_filter=acl_filter, top_k=top_k)
        with open(self.config["paths"]["eval_set"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_SET_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        if pii_answers is not None:
            with open(self.config["paths"]["pii_answers"], "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PII_ANSWER_COLUMNS)
                writer.writeheader()
                writer.writerows(pii_answers)
        args = ["--config", str(self.config_path), "eval", "--run-name", run_name]
        code = main(args)
        assert code == expect_code
        if code != 0:
            return code
        return EvalRun(Path(self.config["paths"]["runs"]) / run_name)

    def gen_eval(self, expect_code=0, **gen_eval):
        """`gen-eval`을 실행하고 평가셋 CSV의 행을 돌려준다.

        `expect_code`가 0이 아니면 그 종료 코드로 끝나는지 확인하고 종료 코드를 돌려준다.
        """
        self.config["gen_eval"].update(gen_eval)
        write_config(self.config_path, self.config)
        code = main(["--config", str(self.config_path), "gen-eval"])
        assert code == expect_code
        if code != 0:
            return code
        with open(self.config["paths"]["eval_set"], newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == EVAL_SET_COLUMNS
            return list(reader)

    def pii_draft(self):
        """`gen-eval`이 쓴 PII 정답셋 초안 CSV의 행."""
        with open(self.config["paths"]["pii_answers"], newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == PII_ANSWER_COLUMNS
            return list(reader)

    def documents(self):
        """문서 표의 행."""
        return pq.read_table(self.config["paths"]["documents"]).to_pylist()

    def _write_config(self, **ask):
        self.config["ask"].update(ask)
        write_config(self.config_path, self.config)

    def chunks(self, doc_id):
        """색인에 들어간 그 문서의 청크 내용, 청크 순서대로."""
        with open_index(Path(self.config["paths"]["index"])) as client:
            points, _ = client.scroll(COLLECTION, limit=10_000)
        found = sorted(
            (p.payload for p in points if p.payload["doc_id"] == doc_id), key=lambda p: p["chunk"]
        )
        return [p["text"] for p in found]

    @property
    def prompts(self):
        """가짜 LLM이 받은 프롬프트(메시지 내용을 이은 글), 받은 순서대로."""
        return ["\n".join(m["content"] for m in r["messages"]) for r in self.vllm.requests]


@pytest.fixture
def make_pipeline(run_collect, tmp_path, monkeypatch, capsys):
    """fixture 칸으로 `collect` → `index`를 실행하고, `ask`를 부르는 `Pipeline`을 돌려준다.

    임베딩은 가짜로 바꾸고, LLM은 가짜 vLLM 서버로 보낸다. NER은 실제 spaCy 모델을 쓴다.
    """
    monkeypatch.setattr(embedding, "load_embedder", lambda config: FakeEmbedder())
    with FakeVLLM() as vllm:

        def make(cells):
            run_collect(cells, quota=2, out="out")
            config_path = tmp_path / "out.yaml"
            config = yaml.safe_load(config_path.read_text())
            config["llm"]["base_url"] = vllm.base_url
            pipeline = Pipeline(config_path, config, vllm, capsys)
            pipeline.index()
            return pipeline

        yield make


@pytest.fixture
def pipeline(make_pipeline):
    """질의응답 fixture(`PIPELINE_CELLS`)로 만든 `Pipeline`."""
    return make_pipeline(PIPELINE_CELLS)
