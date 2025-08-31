from __future__ import annotations

from typing import Dict, Any

import json
import polars as pl


def bins_to_dict(bins: Dict[str, pl.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for var, df in bins.items():
        out[var] = {
            "columns": df.columns,
            "rows": df.to_dicts(),
        }
    return out


def bins_from_dict(obj: Dict[str, Any]) -> Dict[str, pl.DataFrame]:
    bins: Dict[str, pl.DataFrame] = {}
    for var, payload in obj.items():
        rows = payload.get("rows", [])
        bins[var] = pl.DataFrame(rows) if rows else pl.DataFrame(schema={c: pl.Null for c in payload.get("columns", [])})
    return bins


def bins_export_json(bins: Dict[str, pl.DataFrame], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(bins_to_dict(bins), f, ensure_ascii=False, indent=2)


def bins_import_json(path: str) -> Dict[str, pl.DataFrame]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return bins_from_dict(obj)
