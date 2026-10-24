"""평가 하네스: 검수된 평가셋의 질문마다 부서 3개로 질의응답기를 실행하고 지표와 기록을 저장한다.

PII 정답셋의 원문 청크도 지금의 PII 마스커로 마스킹해 PII 지표를 같은 지표 JSON에 넣는다.

실행 결과 폴더(`runs/<실행 이름>/`)는 실행끼리 비교할 때 쓰는 공식 출력이다.
"""

import csv
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from enterprise_rag import embedding, mask, metrics
from enterprise_rag.ask import ask
from enterprise_rag.departments import departments

log = logging.getLogger(__name__)

# 평가셋 CSV의 열. gen-eval(#6)이 이 열로 쓰고, 사람은 keep에 O/X를 적는다.
# doc_ids는 정답 문서 ID이며 여러 개면 `;`로 나눈다.
EVAL_SET_COLUMNS = ["question", "answer", "doc_ids", "department", "question_type", "keep"]
DOC_ID_SEPARATOR = ";"
# PII 정답셋 CSV의 열. gen-eval(#7)이 초안으로 쓰고, 사람은 pii를 고쳐 정답셋으로 확정한다.
# text는 마스킹 전 원문 청크, pii는 "종류: 값"을 `;`로 이은 목록이다(예: "NAME: Kate Morgan; ...").
# split은 개발용(dev)·측정용(test) 표시다(#18). 마스커 규칙은 dev만 보고 만든다.
PII_ANSWER_COLUMNS = ["doc_id", "chunk", "text", "pii", "split"]
PII_SEPARATOR = ";"


class EvalError(Exception):
    """평가를 시작할 수 없을 때. 메시지를 그대로 사용자에게 보여 준다."""


def evaluate(config: dict[str, Any], config_path: Path, run_name: str) -> Path:
    """실행 결과 폴더에 config 사본, 지표 JSON, 실행별 기록 파일을 저장하고 그 폴더를 돌려준다.

    같은 실행 이름의 폴더가 이미 있으면 비교 대상을 덮어쓰지 않도록 실행하기 전에 멈춘다.
    """
    folder = config["paths"]["runs"] / run_name
    if folder.exists():
        raise EvalError(f"실행 결과 폴더가 이미 있다: {folder} (다른 --run-name을 쓴다)")
    questions = load_eval_set(config["paths"]["eval_set"])
    # 사람이 고친 PII 정답셋의 형식 오류는 오래 걸리는 질의응답 실행 전에 알린다
    pii_path = config["paths"]["pii_answers"]
    pii_answers = load_pii_answers(pii_path) if pii_path.exists() else None
    if pii_answers is None:
        log.warning("PII 정답셋이 없어 PII 지표는 비워 둔다: %s", pii_path)
    else:
        check_answers_in_text(pii_answers)
    documents = pq.read_table(
        config["paths"]["documents"], columns=["doc_id", "departments", "text"]
    ).to_pylist()
    allowed = {d["doc_id"]: d["departments"] for d in documents}
    copies = copies_of(documents)
    missing = sorted({i for q in questions for i in q["doc_ids"]} - allowed.keys())
    if missing:
        raise EvalError(f"평가셋의 정답 문서 ID가 문서 표에 없다: {', '.join(missing)}")
    # 모델 불러오기가 첫 실행의 지연 시간에 섞이지 않게 미리 불러 둔다
    embedding.load_embedder(config)
    records = []
    for q in questions:
        for department in departments(config["collect"]["departments"]):
            result = ask(config, department, q["question"])
            record = {
                "question": q["question"],
                "department": department,
                "expected": [
                    {
                        "doc_id": i,
                        "departments": allowed[i],
                        "copies": [{"doc_id": c, "departments": allowed[c]} for c in copies[i]],
                    }
                    for i in q["doc_ids"]
                ],
                "retrieved": [
                    {"doc_id": c["doc_id"], "departments": c["departments"]}
                    for c in result["retrieved"]
                ],
                "answer": result["answer"],
                "latency_ms": result["latency_ms"],
            }
            record["hit_rate"] = metrics.hit_rate(record)
            records.append(record)

    pii_chunks = mask_pii_answers(config, measured_chunks(pii_answers or []))

    folder.mkdir(parents=True)
    shutil.copyfile(config_path, folder / "config.yaml")
    summary = {
        "retrieval_leak_rate": metrics.retrieval_leak_rate(records),
        "refusal_accuracy": metrics.refusal_accuracy(records),
        "hit_rate": metrics.mean_hit_rate(records),
        "pii_miss_rate": metrics.pii_miss_rate(pii_chunks),
        "pii_false_positive_rate": metrics.pii_false_positive_rate(pii_chunks),
        "latency_ms": metrics.latency_summary(records),
    }
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (folder / "records.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("질문 %d개, 실행 %d번의 결과를 %s에 저장했다", len(questions), len(records), folder)
    return folder


def copies_of(documents: list[dict[str, Any]]) -> dict[str, list[str]]:
    """문서 ID마다 그 문서의 사본 ID 목록(자기 자신은 빼고, 없으면 빈 목록).

    사본은 본문의 연속된 공백을 공백 하나로 바꾸고 앞뒤 공백을 지운 뒤 완전히 같은 문서다.
    """
    groups = defaultdict(list)
    for d in documents:
        groups[" ".join(d["text"].split())].append(d["doc_id"])
    return {i: [c for c in group if c != i] for group in groups.values() for i in group}


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """평가셋 CSV에서 `keep`이 O인 행만 읽는다."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["keep"].strip().upper() == "O"]
    for r in rows:
        r["doc_ids"] = [i.strip() for i in r["doc_ids"].split(DOC_ID_SEPARATOR) if i.strip()]
    return rows


def measured_chunks(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PII 지표를 낼 청크: 개발용·측정용 표시가 있으면 측정용(test)만, 없으면(#7 정답셋) 전부."""
    if not any(r.get("split") for r in answers):
        return answers
    return [r for r in answers if r.get("split") == "test"]


def mask_pii_answers(config: dict[str, Any], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PII 정답셋의 청크마다 원문을 지금의 PII 마스커로 마스킹해 지표 계산용 청크로 돌려준다."""
    if not answers:
        return []  # spaCy 모델을 불러오지 않는다
    masker = mask.load_masker(config)
    return [
        {
            "text": row["text"],
            "answers": answer_places(row["text"], [value for _, value in row["pii"]]),
            "masked": [(start, end) for _, start, end in masker.masked_places(row["text"])],
        }
        for row in answers
    ]


def check_answers_in_text(answers: list[dict[str, Any]]) -> None:
    """정답 PII 값을 원문에서 찾지 못하는 행이 있으면 문서 ID, 청크 번호, 값을 알리고 멈춘다.

    찾지 못한 값은 자리가 없어 누락으로도 세지 않으므로, 조용히 넘기지 않는다.
    """
    for row in answers:
        for _, value in row["pii"]:
            if not answer_places(row["text"], [value]):
                raise EvalError(
                    f"PII 정답셋의 값을 원문에서 찾지 못했다: 문서 {row['doc_id']} "
                    f"청크 {row['chunk']}의 {value!r} (원문 글자 그대로 적었는지 확인한다)"
                )


def answer_places(text: str, values: list[str]) -> list[tuple[int, int]]:
    """정답 PII 값들이 원문에서 차지하는 자리(시작, 끝). 앞뒤가 영문자·숫자가 아닌 곳에서만 찾는다.

    예: 값 `Ed`는 "Ed met Eddy"에서 (0, 2) 한 자리다.

    한 값의 자리가 더 긴 값의 자리 안에 들어가면 따로 세지 않는다. 예: 값이 `Denise Hannigan`과
    `Hannigan`이고 원문이 "Denise Hannigan called. Hannigan left."이면 자리는 두 개다.
    """
    places = set()
    for value in set(values):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
        places.update(m.span() for m in re.finditer(pattern, text))
    return sorted(
        p for p in places if not any(q != p and q[0] <= p[0] and p[1] <= q[1] for q in places)
    )


def load_pii_answers(path: Path) -> list[dict[str, Any]]:
    """PII 정답셋 CSV를 읽어 `pii`를 (종류, 값) 목록으로 푼다. 종류나 형식이 틀리면 멈춘다."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for number, r in enumerate(rows, start=2):  # 1행은 머리글
        try:
            r["pii"] = parse_pii(r["pii"])
        except ValueError as e:
            raise EvalError(f"PII 정답셋 {number}행의 PII를 읽을 수 없다: {e}") from None
    return rows


def format_pii(pii: list[tuple[str, str]]) -> str:
    """(종류, 값) 목록을 PII 정답셋 CSV의 `pii` 칸 글로 쓴다. 예: `NAME: Kate Morgan; EMAIL: …`"""
    return f"{PII_SEPARATOR} ".join(f"{kind}: {value}" for kind, value in pii)


def parse_pii(cell: str) -> list[tuple[str, str]]:
    """`format_pii`의 반대. 종류가 없거나 형식이 틀린 항목이 있으면 ValueError."""
    pii = []
    for item in cell.split(PII_SEPARATOR):
        if not item.strip():
            continue
        kind, sep, value = item.partition(":")
        if not sep or kind.strip() not in mask.TYPES or not value.strip():
            raise ValueError(f"{item.strip()!r} ('종류: 값' 형식, 종류는 {', '.join(mask.TYPES)})")
        pii.append((kind.strip(), value.strip()))
    return pii
