"""테스트용 가짜 임베딩과 가짜 vLLM 서버. 같은 입력에 항상 같은 결과를 낸다.

- 가짜 임베딩: 단어를 해시해 단어 빈도 벡터를 만든다. 두 글에 겹치는 단어가 많을수록 가깝다.
  토큰은 공백으로 나눈 단어 하나다.
- 가짜 vLLM: OpenAI 호환 `/v1/chat/completions`를 흉내 낸다. 받은 근거 중 질문과 겹치는 단어가
  있는 근거가 있으면 그 근거 번호를 적어 답하고, 없으면 찾지 못함 응답 문구를 그대로 돌려준다.
  `and`, `the` 같은 불용어는 겹치는 단어로 치지 않는다. `ask`의 프롬프트가 아니면(예: RAGAS)
  빈 JSON `{}`을 돌려준다.
"""

import json
import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NOT_FOUND = "No relevant documents were found."
WORD = re.compile(r"\w+")
STOPWORDS = {"a", "an", "and", "the", "for", "of", "to", "in", "on", "with"}


def words(text: str) -> set[str]:
    return {w.casefold() for w in WORD.findall(text)} - STOPWORDS


class FakeEmbedder:
    dim = 256

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for w in WORD.findall(text):
            vector[zlib.crc32(w.casefold().encode()) % self.dim] += 1.0
        return vector


class FakeVLLM:
    def __init__(self):
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "FakeVLLM":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _answer(self, request: dict) -> str:
        prompt = request["messages"][-1]["content"]
        if "\n\nQuestion: " not in prompt:
            return "{}"
        sources, question = prompt.split("\n\nQuestion: ")
        chunks = re.findall(r"^\[(\d+)\] (.*?)(?=\n\[\d+\] |\Z)", sources, re.M | re.S)
        cited = [n for n, text in chunks if words(text) & words(question)]
        if not cited:
            return NOT_FOUND
        return "The documents cover this " + "".join(f"[{n}]" for n in cited) + "."

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                content = fake._answer(body)
                payload = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        return Handler
