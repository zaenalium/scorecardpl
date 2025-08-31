from __future__ import annotations

from typing import List, Dict, Any

import polars as pl


def write_method_report(rows: List[Dict[str, Any]], out_prefix: str) -> pl.DataFrame:
    """Write a summary report (CSV and Markdown) for method metrics.

    - rows: list of dicts like {"method": str, "auc": float, "ks": float, ...}
    - out_prefix: path prefix, writes `<out_prefix>.csv` and `<out_prefix>.md`.

    Returns the Polars DataFrame created from rows.
    """
    df = pl.DataFrame(rows)
    # order columns: method first if present
    cols = df.columns
    if "method" in cols:
        cols = ["method"] + [c for c in cols if c != "method"]
        df = df.select(cols)
    # write CSV
    df.write_csv(f"{out_prefix}.csv")
    # write Markdown
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        # header
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join([" --- "] * len(cols)) + "|\n")
        for row in df.iter_rows():
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")
    return df

