"""#39 이름 찾기 모델 비교: 마스커의 이름 후보를 찾는 부분만 바꿔 NAME 누락률·오탐률과 속도를 잰다.

보정 규칙(#19·#38), 저자 표기·별명 정규식은 모든 후보에 그대로 둔다. 지표는 #37의 글자 위치 방식이고
NAME만 센다. 속도는 무작위 문서 표본을 한 프로세스로 마스킹해 잰다.

후보:
- lg: 지금 마스커 (spaCy en_core_web_lg)
- trf: spaCy en_core_web_trf (GPU)
- gliner: GLiNER urchade/gliner_medium-v2.1, 레이블 "person" (GPU)
- llm: 지금 띄운 vLLM(Qwen3-8B)에 실명 목록을 물음

실행 (trf, gliner는 프로젝트 의존성에 넣지 않고 실행할 때만 더한다). trf를 GPU로 돌리려면 CuPy와,
CuPy가 찾을 수 있게 프로젝트 환경의 CUDA 라이브러리 경로가 필요하다:
    TRF=<en_core_web_trf-3.8.0 휠 URL: github.com/explosion/spacy-models 릴리스>
    LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cu13/lib \
    uv run --with "en_core_web_trf @ $TRF" --with gliner --with cupy-cuda13x \
        python scripts/compare_name_models.py --split dev --candidates trf \
        --out runs/nm-dev-trf.json
"""

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import spacy
import torch
from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NerModelConfiguration
from presidio_anonymizer import AnonymizerEngine

from enterprise_rag import evaluate, mask, metrics
from enterprise_rag.config import load_config

CANDIDATES = ["lg", "trf", "gliner", "llm"]
GLINER_MODEL = "urchade/gliner_medium-v2.1"
# GLiNER가 "person"으로 볼 최소 점수. 라이브러리 예시의 기본값이다
GLINER_THRESHOLD = 0.5
# GLiNER·LLM이 찾은 자리를 Presidio에 넘길 때의 점수. 마스커는 점수로 거르지 않으므로
# 값 자체는 쓰이지 않는다
FOUND_SCORE = 0.85
# 속도 표본을 고르는 시드와, LLM에 동시에 보낼 문서 수(vLLM이 요청을 함께 처리한다)
SPEED_SEED = 39
LLM_PARALLEL = 8
# LLM에 한 번에 넘기는 글 조각의 최대 길이(글자). 줄 단위로 끊는다 (GLiNER는 mask.GLINER_WINDOW)
LLM_WINDOW = 3000
LLM_PROMPT = (
    "List every real person's name that appears in the text below, exactly as written "
    "(same spelling, capitalization and punctuation). Include full names, first names alone, "
    "surnames alone and author citations like 'Hatsukami DK'. Do not include companies, "
    "products, brands, places, job titles or organizations. Answer with a JSON array of "
    'strings only, for example ["Kate Morgan", "Thed"]. Answer [] if there is none.\n\n'
    "Text:\n"
)


class CandidateMasker(mask.Masker):
    """이름 후보를 찾는 인식기만 바꾼 마스커. 나머지는 `mask.Masker`와 같다."""

    def __init__(self, analyzer: AnalyzerEngine):
        self._analyzer = analyzer
        self._anonymizer = AnonymizerEngine()
        self._entities = ["PERSON"]
        self._operators = {}


class FoundNameRecognizer(EntityRecognizer):
    """`find(text)`가 돌려준 이름 자리에 마스커의 보정 규칙(`_name_pieces`)을 적용한다."""

    def __init__(self, find, name):
        self.find = find
        super().__init__(supported_entities=["PERSON"], name=name)

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for start, end in self.find(text):
            # GLiNER와 LLM은 직함까지 이름으로 돌려준다(`Sen. Mark Warner`). spaCy는 직함을
            # 넣지 않으므로 같은 조건이 되게 직함을 떼고 보정 규칙에 넘긴다. 떼지 않으면
            # `Sen. ` 뒤가 잘려 이름이 버려진다
            if title := mask.TITLE.match(text, start, end):
                start = title.end()
            results += mask._name_pieces(text, RecognizerResult("PERSON", start, end, FOUND_SCORE))
        return results


def build_analyzer(name_recognizer, spacy_model: str, pipes: list[str]) -> AnalyzerEngine:
    """`mask._analyzer`와 같은 구성에서 이름 인식기와 불러올 spaCy 단계만 바꾼다."""
    saved, mask.SPACY_PIPES = mask.SPACY_PIPES, pipes
    try:
        registry = RecognizerRegistry(supported_languages=["en"])
        registry.add_recognizer(name_recognizer)
        patterns = [Pattern("author", mask.AUTHOR, 1.0), Pattern("nickname", mask.NICKNAME, 1.0)]
        registry.add_recognizer(
            PatternRecognizer(
                "PERSON", name="NamePatternRecognizer", patterns=patterns, global_regex_flags=0
            )
        )
        engine = mask.NerOnlyNlpEngine(
            models=[{"lang_code": "en", "model_name": spacy_model}],
            ner_model_configuration=NerModelConfiguration(labels_to_ignore=mask.IGNORED_LABELS),
        )
        engine.load()
        return AnalyzerEngine(registry=registry, nlp_engine=engine, supported_languages=["en"])
    finally:
        mask.SPACY_PIPES = saved


def gliner_finder():
    from gliner import GLiNER

    model = GLiNER.from_pretrained(GLINER_MODEL).to("cuda")

    def find(text):
        spans = []
        for offset, piece in mask._windows(text, mask.GLINER_WINDOW):
            for e in model.predict_entities(piece, ["person"], threshold=GLINER_THRESHOLD):
                spans.append((offset + e["start"], offset + e["end"]))
        return spans

    return find


def llm_finder(config):
    llm = config["llm"]
    client = httpx.Client(base_url=llm["base_url"], timeout=300)

    def ask(piece):
        body = {
            "model": llm["model"],
            "messages": [{"role": "user", "content": LLM_PROMPT + piece}],
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        answer = client.post("/chat/completions", json=body).json()["choices"][0]["message"]
        match = re.search(r"\[.*\]", answer["content"] or "", re.DOTALL)
        try:
            names = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            names = []
        return [n for n in names if isinstance(n, str) and n.strip()]

    def find(text):
        spans = []
        for offset, piece in mask._windows(text, LLM_WINDOW):
            for name in set(ask(piece)):
                pattern = rf"(?<![A-Za-z0-9]){re.escape(name.strip())}(?![A-Za-z0-9])"
                found = re.finditer(pattern, piece)
                spans += [(offset + m.start(), offset + m.end()) for m in found]
        return spans

    return find


def load_masker(kind: str, config):
    if kind == "lg":
        return mask.Masker(config["pii"]["spacy_model"], ["NAME"])
    if kind == "trf":
        spacy.require_gpu()
        recognizer = mask.NameRecognizer(supported_entities=["PERSON"])
        return CandidateMasker(
            build_analyzer(recognizer, "en_core_web_trf", ["transformer", "parser", "ner"])
        )
    find = gliner_finder() if kind == "gliner" else llm_finder(config)
    recognizer = FoundNameRecognizer(find, name=f"{kind}NameRecognizer")
    # 이름은 다른 인식기가 찾으므로 spaCy는 토큰 나누기만 한다
    return CandidateMasker(build_analyzer(recognizer, config["pii"]["spacy_model"], []))


def measure(masker, rows):
    """NAME만으로 누락·오탐을 잰다. 남은 정답과 오탐 값을 함께 돌려준다."""
    chunks, missed, wrong = [], [], []
    started = time.time()
    for row in rows:
        text = row["text"]
        names = [v for kind, v in row["pii"] if kind == "NAME"]
        chunk = {
            "text": text,
            "answers": evaluate.answer_places(text, names),
            "masked": [(s, e) for _, s, e in masker.masked_places(text)],
        }
        chunks.append(chunk)
        key = f"{row['doc_id']}#{row['chunk']}"
        missed += [
            {"chunk": key, "value": text[s:e]}
            for s, e in chunk["answers"]
            if metrics._left_unmasked(chunk, (s, e))
        ]
        wrong += [
            {"chunk": key, "value": text[s:e], "context": text[max(0, s - 50) : e + 50]}
            for s, e in chunk["masked"]
            if not any(metrics._overlaps((s, e), a) for a in chunk["answers"])
        ]
    return {
        "miss_rate": metrics.pii_miss_rate(chunks),
        "false_positive_rate": metrics.pii_false_positive_rate(chunks),
        "answers": sum(len(c["answers"]) for c in chunks),
        "masked": sum(len(c["masked"]) for c in chunks),
        "missed": missed,
        "false_positives": wrong,
        "seconds": round(time.time() - started, 1),
    }


def speed(masker, documents, sample, parallel):
    """표본 문서를 마스킹한 시간으로 9,000건 전체 시간을 글자 수 비례로 추정한다(분)."""
    picked = random.Random(SPEED_SEED).sample(documents, sample)
    texts = [d["title"] + "\n" + d["text"] for d in picked]
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    if parallel == 1:  # spaCy의 GPU 설정은 스레드마다 따로라, 한 프로세스는 이 스레드에서 돌린다
        for text in texts:
            masker.masked_places(text)
    else:
        with ThreadPoolExecutor(parallel) as pool:
            list(pool.map(masker.masked_places, texts))
    seconds = time.time() - started
    total_chars = sum(len(d["title"]) + 1 + len(d["text"]) for d in documents)
    return {
        "sample_docs": sample,
        "sample_chars": sum(map(len, texts)),
        "sample_seconds": round(seconds, 1),
        "estimated_minutes_9000": round(seconds * total_chars / sum(map(len, texts)) / 60, 1),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--speed-docs", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    rows = [
        r
        for r in evaluate.load_pii_answers(config["paths"]["pii_answers"])
        if r["split"] == args.split
    ]
    documents = pq.read_table(config["paths"]["documents"], columns=["title", "text"]).to_pylist()
    results = {}
    for kind in args.candidates.split(","):
        masker = load_masker(kind, config)
        result = measure(masker, rows)
        # LLM은 서버가 요청을 함께 처리하므로 문서 8개를 동시에 보낸다. 나머지는 한 프로세스로 잰다
        result["speed"] = speed(
            masker, documents, args.speed_docs, LLM_PARALLEL if kind == "llm" else 1
        )
        results[kind] = result
        print(
            kind,
            f"누락 {result['miss_rate']} 오탐 {result['false_positive_rate']}",
            f"(정답 자리 {result['answers']}, 가린 자리 {result['masked']})",
            f"9,000건 추정 {result['speed']['estimated_minutes_9000']}분",
            flush=True,
        )
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
