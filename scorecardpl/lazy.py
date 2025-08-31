from __future__ import annotations

from typing import Dict, Optional, Sequence

import polars as pl
import numpy as np


def woebin_ply_lazy(lf: pl.LazyFrame, bins: Dict[str, pl.DataFrame], keep_bins: bool = True) -> pl.LazyFrame:
    out = lf
    drop_cols: list[str] = []
    for var, bdf in bins.items():
        cols = bdf.columns
        if "lower" in cols and "upper" in cols:
            bdf_sorted = bdf.sort("ord") if "ord" in cols else bdf
            edges = [float(x) for x in bdf_sorted.select("lower").to_series().to_list()] + [
                float(bdf_sorted.select("upper").to_series().to_list()[-1])
            ]
            labels = bdf_sorted.select("bin").to_series().to_list()
            out = out.with_columns(
                pl.cut(pl.col(var), bins=edges, labels=labels).cast(pl.Utf8).alias(f"{var}_bin")
            )
        else:
            if 'levels' in cols:
                mapping = bdf.select([pl.col('bin'), pl.col('levels')]).explode('levels').rename({'levels': 'raw'})
                out = out.with_columns(
                    pl.when(pl.col(var).is_null()).then(pl.lit('__NA__')).otherwise(pl.col(var).cast(pl.Utf8)).alias(f"__{var}_raw")
                )
                out = out.join(mapping.lazy(), left_on=f"__{var}_raw", right_on='raw', how='left')
                out = out.with_columns(
                    pl.when(pl.col('bin').is_null()).then(pl.lit('__OTHER__')).otherwise(pl.col('bin')).alias(f"{var}_bin")
                ).drop(['raw','bin', f"__{var}_raw"])
            else:
                valid_bins = bdf.select("bin").to_series().to_list()
                out = out.with_columns(
                    pl.when(pl.col(var).is_null())
                      .then(pl.lit("__NA__"))
                      .when(pl.col(var).cast(pl.Utf8).is_in(valid_bins))
                      .then(pl.col(var).cast(pl.Utf8))
                      .otherwise(pl.lit("__OTHER__"))
                      .alias(f"{var}_bin")
                )

        # join WOE
        wmap = bdf.select([pl.col("bin").cast(pl.Utf8), pl.col("woe")]).rename({"bin": f"{var}_bin", "woe": f"{var}_woe"})
        out = out.join(wmap.lazy(), on=f"{var}_bin", how="left")
        out = out.with_columns(pl.col(f"{var}_woe").fill_null(0.0))
        if not keep_bins:
            drop_cols.append(f"{var}_bin")
    if drop_cols:
        out = out.drop(drop_cols)
    return out


def score_lazy(lf: pl.LazyFrame, points_map: Dict[str, pl.DataFrame], score_col: str = 'score', prefer_woe: bool = True) -> pl.LazyFrame:
    exprs = []
    bias = 0.0
    if '__INTERCEPT__' in points_map:
        bias = float(points_map['__INTERCEPT__'].select('points').item())
    expr = pl.lit(bias)
    for var, pm in points_map.items():
        if var == '__INTERCEPT__':
            continue
        if prefer_woe:
            df_pm = pm.filter(pl.col('woe') != 0.0)
            if df_pm.height == 0:
                continue
            k = float((df_pm.select((pl.col('points')/pl.col('woe')).alias('ratio')).select(pl.col('ratio').median()).to_series().item()))
            expr = expr + (pl.col(f"{var}_woe").fill_null(0.0) * k)
        else:
            m = pm.rename({'bin': f"{var}_bin", 'points': f"{var}_pts"})
            lf = lf.join(m.lazy().select([f"{var}_bin", f"{var}_pts"]), on=f"{var}_bin", how='left')
            expr = expr + pl.col(f"{var}_pts").fill_null(0.0)
    return lf.with_columns(expr.alias(score_col))


def scan_parquet_select(paths: Sequence[str] | str, columns: Optional[Sequence[str]] = None) -> pl.LazyFrame:
    return pl.scan_parquet(paths).select(columns) if columns else pl.scan_parquet(paths)


def sink_parquet_safe(lf: pl.LazyFrame, path: str) -> None:
    try:
        lf.sink_parquet(path)
    except Exception:
        # fallback
        lf.collect(streaming=True).write_parquet(path)

