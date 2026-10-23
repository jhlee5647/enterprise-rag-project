"""`collect` → `index` → `ask`: 색인 위치가 서버 주소면 Qdrant 서버에, 폴더면 로컬 모드로 연다.

테스트 이름의 `tNN`은 티켓 #22 Acceptance criteria의 사례 번호다. `extra`는 구현 중 추가한 사례다.
자동 테스트에는 Qdrant 서버가 없으므로 서버에 붙는 경로는 서버가 없을 때만 확인한다.
T1(색인 위치가 폴더면 지금처럼 동작)은 로컬 모드로 도는 기존 테스트 전체로 확인한다.
"""

from enterprise_rag.cli import main
from enterprise_rag.index import open_index
from tests.conftest import write_config

NO_SERVER = "http://127.0.0.1:9"
QUESTION = "influencer budget"


def test_t02_missing_server_stops_index_and_ask_with_address_and_how_to_start(pipeline, capsys):
    pipeline.config["paths"]["index"] = NO_SERVER
    write_config(pipeline.config_path, pipeline.config)

    for args in (["index"], ["ask", "--dept", "marketing", QUESTION]):
        capsys.readouterr()
        code = main(["--config", str(pipeline.config_path), *args])

        assert code != 0
        err = capsys.readouterr().err
        assert NO_SERVER in err
        assert "scripts/qdrant-server.sh start" in err


def test_t03_index_and_ask_use_configured_collection(pipeline):
    pipeline.config["index"]["collection"] = "chunks-nomask"
    pipeline.index()

    asked = pipeline("marketing", QUESTION)

    assert "pnot0001" in asked.retrieved_ids
    with open_index(pipeline.config["paths"]["index"]) as client:
        assert client.collection_exists("chunks-nomask")
        # 처음 색인(fixture)이 만든 기본 컬렉션은 그대로 남는다
        assert client.collection_exists("chunks")
