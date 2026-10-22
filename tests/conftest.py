import pyarrow.parquet as pq
import pytest
import yaml

from enterprise_rag.cli import main
from tests.fake_idl import FakeIDL


class Collected:
    """`collect` 실행 결과: 문서 표의 행과 문서 ID 목록."""

    def __init__(self, documents_path, sample_ids_path):
        self.table = pq.read_table(documents_path)
        self.rows = self.table.to_pylist()
        self.by_id = {r["doc_id"]: r for r in self.rows}
        self.sample_ids = sample_ids_path.read_text().split()


@pytest.fixture
def run_collect(tmp_path):
    """가짜 IDL 서버를 띄우고 config를 써서 `collect`를 실행한다."""

    def run(cells, quota=2, seed=42, out="out"):
        out_dir = tmp_path / out
        config = {
            "paths": {
                "documents": str(out_dir / "documents.parquet"),
                "sample_ids": str(out_dir / "sample_ids.txt"),
            },
            "collect": {
                "seed": seed,
                "quota": quota,
                "doc_types": ["email", "document", "presentation"],
                "excluded_types": ["publication", "article", "news", "transcript", "video", "audio"],
                "departments": {
                    "Marketing": "marketing",
                    "Government & Public Affairs": "regulatory",
                    "Product Design & Quality": "product",
                },
            },
        }
        with FakeIDL(cells) as idl:
            config["idl"] = {"solr_url": idl.solr_url, "s3_url": idl.s3_url}
            config_path = tmp_path / f"{out}.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
            assert main(["--config", str(config_path), "collect"]) == 0
        result = Collected(out_dir / "documents.parquet", out_dir / "sample_ids.txt")
        result.solr_requests = idl.solr_requests
        return result

    return run
