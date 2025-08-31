from __future__ import annotations

from typing import Dict, Any

import polars as pl

from .utils import to_pl_df, cut_expr


def woebin_ply(df: Any, bins: Dict[str, pl.DataFrame], keep_bins: bool = True) -> pl.DataFrame:
    """Apply WOE bins to transform variables into `<var>_bin` and `<var>_woe` columns (Polars).

    - Numeric variables: uses stored [lower, upper] edges via `pl.cut`.
    - Categorical variables: maps unseen values to `__OTHER__`, nulls to `__NA__`.
    """
    out = to_pl_df(df)
    for var, bdf in bins.items():
        cols = bdf.columns
        if "lower" in cols and "upper" in cols:
            # numeric
            bdf_sorted = bdf.sort("ord") if "ord" in cols else bdf
            edges = [float(x) for x in bdf_sorted.select("lower").to_series().to_list()] + [
                float(bdf_sorted.select("upper").to_series().to_list()[-1])
            ]
            labels = bdf_sorted.select("bin").to_series().to_list()
            out = out.with_columns(
                cut_expr(pl.col(var), edges, labels).alias(f"{var}_bin")
            )
        else:
            # categorical
            if 'levels' in cols:
                # build raw -> bin mapping with names scoped to var to avoid collisions
                mapping = (
                    bdf
                    .select([pl.col('bin'), pl.col('levels')])
                    .explode('levels')
                    .rename({'levels': f'__{var}_raw_key', 'bin': f'__{var}_bin_map'})
                )
                # handle nulls and unseen
                out = out.with_columns(
                    pl.when(pl.col(var).is_null()).then(pl.lit('__NA__')).otherwise(pl.col(var).cast(pl.Utf8)).alias(f"__{var}_raw")
                )
                out = out.join(mapping, left_on=f"__{var}_raw", right_on=f'__{var}_raw_key', how='left')
                out = out.with_columns(
                    pl.when(pl.col(f'__{var}_bin_map').is_null()).then(pl.lit('__OTHER__')).otherwise(pl.col(f'__{var}_bin_map')).alias(f"{var}_bin")
                )
                drop_cols = [f"__{var}_raw", f'__{var}_raw_key', f'__{var}_bin_map']
                out = out.drop([c for c in drop_cols if c in out.columns])
            else:
                valid_bins = set(bdf.select("bin").to_series().to_list())
                out = out.with_columns(
                    pl.when(pl.col(var).is_null())
                      .then(pl.lit("__NA__"))
                      .when(pl.col(var).cast(pl.Utf8).is_in(list(valid_bins)))
                      .then(pl.col(var).cast(pl.Utf8))
                      .otherwise(pl.lit("__OTHER__"))
                      .alias(f"{var}_bin")
                )

        # attach WOE by joining on bin label
        wmap = bdf.select([pl.col("bin").cast(pl.Utf8), pl.col("woe")]).rename({"bin": f"{var}_bin", "woe": f"{var}_woe"})
        out = out.join(wmap, on=f"{var}_bin", how="left")
        out = out.with_columns(pl.col(f"{var}_woe").fill_null(0.0))
        if not keep_bins:
            out = out.drop(f"{var}_bin")
    return out
