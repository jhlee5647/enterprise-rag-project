"""CLI 진입점. 명령 다섯 개를 받는다."""

import argparse
import logging
import sys
from pathlib import Path

from enterprise_rag.collect import collect
from enterprise_rag.config import load_config

COMMANDS = {
    "collect": "IDL에서 표본을 뽑아 본문을 받고 허용 부서를 계산해 문서 표로 저장한다",
    "index": "문서 표의 본문에서 PII를 가리고 청크로 잘라 색인한다 (미구현)",
    "ask": "부서를 지정해 질문하고 출처가 붙은 답을 받는다 (미구현)",
    "gen-eval": "평가셋 초안과 PII 정답셋 초안을 만든다 (미구현)",
    "eval": "평가셋으로 시스템을 평가하고 지표를 저장한다 (미구현)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="JUUL 사내 문서 RAG PoC")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="config 파일 경로")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 요청마다 찍히는 로그를 끈다

    if args.command == "collect":
        collect(load_config(args.config))
        return 0

    print(f"`{args.command}`은 아직 구현되지 않았다.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
