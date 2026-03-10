import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    target_dir = Path(path).parent
    with NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8", dir=target_dir) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
