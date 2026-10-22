"""임베딩 모델: 청킹에 쓸 토큰 위치와 임베딩을 준다. 앱 프로세스 안에서 실행한다."""

from typing import Any, Protocol


class Embedder(Protocol):
    dim: int

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """토큰마다 본문 안의 (시작, 끝) 글자 위치."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def load_embedder(config: dict[str, Any]) -> Embedder:
    return SentenceTransformerEmbedder(config["embedding"]["model"])


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # 무거워서 쓸 때만 불러온다

        self._model = SentenceTransformer(model)
        self.dim = self._model.get_embedding_dimension()

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [tuple(span) for span in encoded["offset_mapping"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        # Qwen3-Embedding은 질문에 검색용 지시문(prompt_name="query")을 붙여야 한다.
        prompt_name = "query" if "query" in self._model.prompts else None
        return self._model.encode(text, prompt_name=prompt_name, normalize_embeddings=True).tolist()
