"""config 파일 읽기. 상대 경로는 config 파일이 있는 폴더 기준으로 푼다."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    base = path.resolve().parent
    config["paths"] = {k: base / v for k, v in config["paths"].items()}
    return config
