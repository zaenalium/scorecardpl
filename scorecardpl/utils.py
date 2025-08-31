from __future__ import annotations

from typing import Optional, Sequence, Tuple, List, Any

import numpy as np
import polars as pl
import pandas as pd


def to_pl_df(df: Any) -> pl.DataFrame:
    """Convert pandas.DataFrame or Polars DataFrame to Polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError("Expected a Polars or pandas DataFrame")


def to_pl_series(s: Any, name: Optional[str] = None) -> pl.Series:
    """Convert pandas.Series or Polars Series to Polars Series."""
    if isinstance(s, pl.Series):
        return s
    if isinstance(s, pd.Series):
        return pl.from_pandas(s.to_frame(name or s.name)).to_series()
    # numpy array fallback
    if isinstance(s, (list, tuple, np.ndarray)):
        return pl.Series(name or "series", s)
    raise TypeError("Expected a Polars/pandas Series or array-like")


def to_pl_lf(df: Any) -> pl.LazyFrame:
    """Convert pandas/Polars DataFrame or LazyFrame to Polars LazyFrame.

    Note: For pandas input this materializes into memory before converting.
    For big data, prefer using `pl.scan_parquet` to create a LazyFrame.
    """
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    raise TypeError("Expected a Polars LazyFrame/DataFrame or pandas DataFrame")


def split_df(df: Any, y: str, test_size: float = 0.3, random_state: Optional[int] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Stratified train/valid split using Polars.

    - df: Polars DataFrame containing binary target column `y` (0/1)
    - test_size: fraction for the validation split
    - random_state: seed
    """
    df = to_pl_df(df)
    if y not in df.columns:
        raise ValueError(f"Target column {y} not found")
    parts = df.partition_by(y, maintain_order=True)
    trains: List[pl.DataFrame] = []
    valids: List[pl.DataFrame] = []
    for g in parts:
        n = g.height
        n_valid = int(round(n * test_size))
        if n_valid <= 0:
            trains.append(g)
            continue
        g_shuf = g.sample(n=n, shuffle=True, seed=random_state)
        valids.append(g_shuf.head(n_valid))
        trains.append(g_shuf.slice(n_valid))
    train = pl.concat(trains, how="vertical")
    valid = pl.concat(valids, how="vertical") if valids else pl.DataFrame(schema=df.schema)
    return train, valid


def _missing_rate(df: pl.DataFrame, col: str) -> float:
    return float(df.select(pl.col(col).is_null().mean()).item())


def _unique_count(df: pl.DataFrame, col: str) -> int:
    return int(df.select(pl.col(col).n_unique()).item())


def var_filter(df: Any, y: str, x: Optional[Sequence[str]] = None,
               miss_thres: float = 0.95, unique_thres: int = 1) -> pl.DataFrame:
    """Remove variables with too-high missing rate or too-low uniqueness (Polars)."""
    df = to_pl_df(df)
    if x is None:
        x = [c for c in df.columns if c != y]
    keep = []
    for c in x:
        mr = _missing_rate(df, c)
        uc = _unique_count(df, c)
        if mr < miss_thres and uc > unique_thres:
            keep.append(c)
    cols = [y] + keep
    return df.select(cols)


def ensure_binary_target(s: Any) -> pl.Series:
    s = to_pl_series(s)
    vals = set(s.drop_nulls().unique().to_list())
    if vals.issubset({0, 1}):
        return s.cast(pl.Int64)
    raise ValueError("Target must be binary 0/1")


def cut_expr(expr: pl.Expr, edges: Sequence[float], labels: Sequence[str]) -> pl.Expr:
    """Polars expression that bins numeric values into interval labels.

    Intervals follow (edges[i], edges[i+1]] semantics. Returns Utf8.
    """
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must be len(edges)-1")
    # Build chained when-then for each interval
    lower = float(edges[0])
    upper = float(edges[1])
    cond = pl.when((expr > lower) & (expr <= upper)).then(pl.lit(labels[0]))
    for i in range(1, len(labels)):
        lower = float(edges[i])
        upper = float(edges[i + 1])
        cond = cond.when((expr > lower) & (expr <= upper)).then(pl.lit(labels[i]))
    return cond.otherwise(None).cast(pl.Utf8)
