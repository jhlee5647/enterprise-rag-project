"""CLI 진입점. 명령 다섯 개를 받는다."""

import argparse
import json
import logging
import sys
from pathlib import Path

from enterprise_rag.ask import ask
from enterprise_rag.collect import collect
from enterprise_rag.config import load_config
from enterprise_rag.departments import departments
from enterprise_rag.evaluate import EvalError, evaluate
from enterprise_rag.index import index

COMMANDS = {
    "collect": "IDL에서 표본을 뽑아 본문을 받고 허용 부서를 계산해 문서 표로 저장한다",
    "index": "문서 표의 본문과 제목을 PII 마스킹하고 청크로 잘라 색인한다",
    "ask": "부서를 지정해 질문하고 출처가 붙은 답을 받는다",
    "gen-eval": "평가셋 초안과 PII 정답셋 초안을 만든다 (미구현)",
    "eval": "검수된 평가셋으로 권한·검색 지표를 재고 실행 결과 폴더에 저장한다",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="JUUL 사내 문서 RAG PoC")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="config 파일 경로")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {name: sub.add_parser(name, help=help_text) for name, help_text in COMMANDS.items()}
    commands["ask"].add_argument("--dept", required=True, help="질문하는 사람의 부서")
    commands["ask"].add_argument("question", help="영어 질문")
    commands["eval"].add_argument(
        "--run-name", required=True, help="실행 이름. 결과는 runs/<실행 이름>/에 저장한다"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 요청마다 찍히는 로그를 끈다

    if args.command == "collect":
        collect(load_config(args.config))
        return 0
    if args.command == "index":
        index(load_config(args.config))
        return 0
    if args.command == "ask":
        config = load_config(args.config)
        valid = departments(config["collect"]["departments"])
        if args.dept not in valid:
            print(
                f"알 수 없는 부서: {args.dept} (쓸 수 있는 부서: {', '.join(valid)})",
                file=sys.stderr,
            )
            return 2
        result = ask(config, args.dept, args.question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval":
        try:
            evaluate(load_config(args.config), args.config, args.run_name)
        except EvalError as e:
            print(e, file=sys.stderr)
            return 2
        return 0

    print(f"`{args.command}`은 아직 구현되지 않았다.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
