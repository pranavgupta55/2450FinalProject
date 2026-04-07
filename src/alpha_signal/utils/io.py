from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    df.to_csv(resolved, index=False)
