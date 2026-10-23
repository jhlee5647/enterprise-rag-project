"""config 파일 읽기. 상대 경로는 config 파일이 있는 폴더 기준으로 푼다. 서버 주소는 그대로 둔다."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    base = path.resolve().parent
    config["paths"] = {k: v if is_url(v) else base / v for k, v in config["paths"].items()}
    return config


def is_url(value: str | Path) -> bool:
    """`http://`, `https://`로 시작하는 서버 주소인지. 예: 색인 위치가 Qdrant 서버일 때."""
    return str(value).startswith(("http://", "https://"))
