"""저장된 응답 fixture를 돌려주는 가짜 IDL Solr·S3 서버.

칸(IDL 토픽, 문서 종류)마다 후보 문서 ID를 정해 두면, Solr 요청의 `q`에서 칸을 읽어
그 후보들을 실제 Solr 응답 형식으로 돌려준다. 후보 순서가 곧 `sort=random_<seed>` 순서다.
실제 Solr처럼 `rows`를 무시하고 한 페이지씩 돌려주며, 페이징을 거치도록 페이지 크기를 작게 둔다.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES = Path(__file__).parent / "fixtures" / "idl"
SOLR_PATH = "/solr/ltdl3/query"
PAGE_SIZE = 2


class FakeIDL:
    def __init__(self, cells: dict[tuple[str, str], list[str]]):
        self.cells = cells
        self.docs = {d["id"]: d for d in json.loads((FIXTURES / "solr_docs.json").read_text())}
        self.solr_requests: list[dict[str, str]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def solr_url(self) -> str:
        return self.base_url + SOLR_PATH

    @property
    def s3_url(self) -> str:
        return self.base_url + "/s3"

    def __enter__(self) -> "FakeIDL":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _solr(self, params: dict[str, str]) -> dict:
        self.solr_requests.append(params)
        q = params["q"]
        doc_type = re.search(r"(?<!NOT )type:(\w+)", q).group(1)
        topic = re.search(r'topic:"([^"]+)"', q).group(1)
        ids = self.cells.get((topic, doc_type), [])
        start = int(params.get("start", 0))
        page = [self.docs[i] for i in ids[start : start + PAGE_SIZE]]
        return {
            "responseHeader": {"status": 0, "params": params},
            "response": {"numFound": len(ids), "start": start, "docs": page},
        }

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)
                if url.path == SOLR_PATH:
                    params = {k: v[0] for k, v in parse_qs(url.query).items()}
                    self._send(200, json.dumps(fake._solr(params)).encode())
                    return
                # /s3/a/b/c/d/<id>/<id>.ocr
                body = FIXTURES / "s3" / Path(url.path).name
                expected = "/s3/" + "/".join(body.stem[:4]) + f"/{body.stem}/{body.name}"
                if url.path == expected and body.is_file():
                    self._send(200, body.read_bytes())
                else:
                    self._send(404, b"<Error><Code>NoSuchKey</Code></Error>")

            def _send(self, status: int, payload: bytes):
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        return Handler
