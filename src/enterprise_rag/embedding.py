"""임베딩 모델: 청킹에 쓸 토큰 위치와 임베딩을 준다. 하이브리드 검색에 쓸 BM25 단어 가중치도 준다.
모두 앱 프로세스 안에서 실행한다."""

from functools import cache
from typing import Any, Protocol


class Embedder(Protocol):
    dim: int

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """토큰마다 본문 안의 (시작, 끝) 글자 위치."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# BM25 단어 가중치 모델(fastembed). IDF는 Qdrant가 서버에서 계산한다 (#52)
BM25_MODEL = "Qdrant/bm25"

# 임베딩 모델을 불러올 정밀도. 문서와 질문 모두 이 정밀도로 임베딩한다
DTYPES = ["bfloat16", "float32"]


def load_embedder(config: dict[str, Any]) -> Embedder:
    """config의 임베딩 모델을 config의 정밀도(`embedding.dtype`)로 불러온다.

    정밀도가 `DTYPES`에 없으면 ValueError.
    """
    dtype = config["embedding"]["dtype"]
    check_dtype(dtype)
    return load_model(config["embedding"]["model"], dtype)


def check_dtype(dtype: str) -> None:
    """정밀도가 `DTYPES`에 없으면 ValueError."""
    if dtype not in DTYPES:
        raise ValueError(
            f"embedding.dtype(임베딩 정밀도)는 {', '.join(DTYPES)} 중 하나여야 한다: {dtype}"
        )


@cache
def load_model(model: str, dtype: str) -> "SentenceTransformerEmbedder":
    """임베딩 모델을 불러온다. 테스트는 이 함수를 가짜 임베딩으로 바꾼다.

    같은 모델은 한 번만 불러온다. `eval`이 질문·부서마다 `ask`를 부르기 때문이다.
    """
    return SentenceTransformerEmbedder(model, dtype)


class SentenceTransformerEmbedder:
    def __init__(self, model: str, dtype: str):
        import torch  # 무거워서 쓸 때만 불러온다
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, model_kwargs={"dtype": getattr(torch, dtype)})
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


@cache
def load_bm25() -> Any:
    """BM25 단어 가중치 모델을 불러온다. 같은 프로세스에서는 한 번만 불러온다."""
    from fastembed import SparseTextEmbedding  # 무거워서 쓸 때만 불러온다

    return SparseTextEmbedding(BM25_MODEL)


def bm25_documents(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """청크마다 BM25 sparse 벡터(단어 번호, 가중치)."""
    return [(e.indices.tolist(), e.values.tolist()) for e in load_bm25().embed(texts)]


def bm25_query(text: str) -> tuple[list[int], list[float]]:
    """질문의 BM25 sparse 벡터(단어 번호, 가중치)."""
    e = next(iter(load_bm25().query_embed(text)))
    return e.indices.tolist(), e.values.tolist()
