"""평가 하네스: 검수된 평가셋의 질문마다 부서 3개로 질의응답기를 실행하고 지표와 기록을 저장한다.

실행 결과 폴더(`runs/<실행 이름>/`)는 실행끼리 비교할 때 쓰는 공식 출력이다.
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from enterprise_rag import metrics
from enterprise_rag.ask import ask
from enterprise_rag.departments import departments

log = logging.getLogger(__name__)

# 평가셋 CSV의 열. gen-eval(#6)이 이 열로 쓰고, 사람은 keep에 O/X를 적는다.
# doc_ids는 정답 문서 ID이며 여러 개면 `;`로 나눈다.
EVAL_SET_COLUMNS = ["question", "answer", "doc_ids", "department", "question_type", "keep"]
DOC_ID_SEPARATOR = ";"


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
    allowed = {
        d["doc_id"]: d["departments"]
        for d in pq.read_table(
            config["paths"]["documents"], columns=["doc_id", "departments"]
        ).to_pylist()
    }
    missing = sorted({i for q in questions for i in q["doc_ids"]} - allowed.keys())
    if missing:
        raise EvalError(f"평가셋의 정답 문서 ID가 문서 표에 없다: {', '.join(missing)}")
    records = []
    for q in questions:
        for department in departments(config["collect"]["departments"]):
            result = ask(config, department, q["question"])
            record = {
                "question": q["question"],
                "department": department,
                "expected": [{"doc_id": i, "departments": allowed[i]} for i in q["doc_ids"]],
                "retrieved": [
                    {"doc_id": c["doc_id"], "departments": c["departments"]}
                    for c in result["retrieved"]
                ],
                "answer": result["answer"],
            }
            record["hit_rate"] = metrics.hit_rate(record)
            records.append(record)

    folder.mkdir(parents=True)
    shutil.copyfile(config_path, folder / "config.yaml")
    summary = {
        "retrieval_leak_rate": metrics.retrieval_leak_rate(records),
        "refusal_accuracy": metrics.refusal_accuracy(records),
        "hit_rate": metrics.mean_hit_rate(records),
    }
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (folder / "records.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("질문 %d개, 실행 %d번의 결과를 %s에 저장했다", len(questions), len(records), folder)
    return folder


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """평가셋 CSV에서 `keep`이 O인 행만 읽는다."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["keep"].strip().upper() == "O"]
    for r in rows:
        r["doc_ids"] = [i.strip() for i in r["doc_ids"].split(DOC_ID_SEPARATOR) if i.strip()]
    return rows
