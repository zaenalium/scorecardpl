from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import polars as pl

from .transform import woebin_ply
from .utils import to_pl_df


def iv_summary(bins: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Summarize total IV per variable from bins.

    Returns pl.DataFrame with columns: variable, iv, nbin
    """
    rows = []
    for var, df in bins.items():
        if 'iv' not in df.columns:
            continue
        iv_total = float(df.select(pl.col('iv').sum()).item())
        rows.append({"variable": var, "iv": iv_total, "nbin": df.height})
    if not rows:
        return pl.DataFrame({"variable": [], "iv": [], "nbin": []})
    return pl.DataFrame(rows).sort('iv', descending=True)


def var_filter_by_iv(bins: Dict[str, pl.DataFrame], min_iv: float = 0.02, max_iv: float = np.inf) -> List[str]:
    """Return variable names whose total IV is within [min_iv, max_iv]."""
    summary = iv_summary(bins)
    if summary.height == 0:
        return []
    df = summary.filter((pl.col('iv') >= min_iv) & (pl.col('iv') <= max_iv))
    return df.select('variable').to_series().to_list()


def _psi_from_counts(ref_counts: pl.DataFrame, cmp_counts: pl.DataFrame, eps: float = 1e-9) -> float:
    # both DataFrames have columns: bin, cnt
    ref_total = max(ref_counts.select(pl.col('cnt').sum()).item(), eps)
    cmp_total = max(cmp_counts.select(pl.col('cnt').sum()).item(), eps)
    joined = (
        ref_counts.rename({'cnt': 'ref_cnt'})
        .join(cmp_counts.rename({'cnt': 'cmp_cnt'}), on='bin', how='outer')
        .with_columns([
            pl.col('ref_cnt').fill_null(0.0),
            pl.col('cmp_cnt').fill_null(0.0),
        ])
        .with_columns([
            (pl.col('ref_cnt') / ref_total).alias('ref_p'),
            (pl.col('cmp_cnt') / cmp_total).alias('cmp_p'),
        ])
        .with_columns(
            ((pl.col('ref_p') - pl.col('cmp_p')) * ((pl.col('ref_p') + eps) / (pl.col('cmp_p') + eps)).log()).alias('psi')
        )
    )
    return float(joined.select(pl.col('psi').sum()).item())


def psi(df_ref: Any, df_cmp: Any, bins: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Compute PSI between reference and comparison datasets per variable using `bins`.

    Returns pl.DataFrame with columns: variable, psi
    """
    # ensure bin columns exist
    ref_b = woebin_ply(to_pl_df(df_ref), bins)
    cmp_b = woebin_ply(to_pl_df(df_cmp), bins)

    rows = []
    for var, bdf in bins.items():
        bin_col = f"{var}_bin"
        if bin_col not in ref_b.columns:
            continue
        labels = bdf.select('bin').to_series().to_list()
        # get counts per label, ensure all labels present
        ref_counts = (
            ref_b.group_by(bin_col).len().rename({bin_col: 'bin', 'len': 'cnt'})
            .join(pl.DataFrame({'bin': labels}), on='bin', how='outer')
            .with_columns(pl.col('cnt').fill_null(0.0))
        )
        cmp_counts = (
            cmp_b.group_by(bin_col).len().rename({bin_col: 'bin', 'len': 'cnt'})
            .join(pl.DataFrame({'bin': labels}), on='bin', how='outer')
            .with_columns(pl.col('cnt').fill_null(0.0))
        )
        val = _psi_from_counts(ref_counts, cmp_counts)
        rows.append({"variable": var, "psi": val})
    return pl.DataFrame(rows).sort('psi', descending=True)
