"""
Big data pipeline example (~100M rows) using partitioned Parquet files.

Steps:
- Build bins from a sampled subset
- Train a streaming SGD scorecard via partial_fit over files
- Transform/score each file and write scores to Parquet

Adjust INPUT_GLOB, TARGET, FEATURES, and OUTPUT_DIR to your environment.
"""

import glob
import os
from typing import List

import polars as pl

from scorecardpl import (
    woebin, woebin_ply, scorecard_sgd, scorecard_ply,
    iter_parquet, iter_parquet_row_groups, perf_eva,
    woebin_ply_lazy, score_lazy, scan_parquet_select, sink_parquet_safe
)


INPUT_GLOB = "data/*.parquet"  # change to your path pattern
OUTPUT_DIR = "examples/out_scores"
TARGET = "y"
FEATURES: List[str] = []  # leave empty to infer from first file (all non-target)


def sample_for_binning(files: List[str], cols: List[str], sample_rows: int = 2_000_000, seed: int = 42) -> pl.DataFrame:
    out = []
    total = 0
    for path in files:
        n_left = sample_rows - total
        if n_left <= 0:
            break
        # read up to n_left rows from this file
        df = pl.read_parquet(path, columns=cols)
        if df.height > n_left:
            df = df.sample(n=n_left, shuffle=True, seed=seed)
        out.append(df)
        total += df.height
    return pl.concat(out) if out else pl.DataFrame()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        raise SystemExit(f"No files matched: {INPUT_GLOB}")

    # Infer columns
    first = pl.read_parquet(files[0])
    global FEATURES
    if not FEATURES:
        FEATURES = [c for c in first.columns if c != TARGET]
    cols = [TARGET] + FEATURES

    # 1) Build bins from a sample (edges from sample; WOE/IV computed on full data is optional)
    sample_df = sample_for_binning(files, cols, sample_rows=1_000_000)
    bins = woebin(
        sample_df,
        y=TARGET,
        x=FEATURES,
        bins=6,
        method="chi2",
        chi2_params={"init_bins": 60},
        monotonic="auto",
        cat_max_bins=5,
        cat_method="target_rate",
    )

    # 2) Train streaming SGD on WOE features via partial_fit over row groups
    sc_online = scorecard_sgd(bins, y=TARGET, frames=iter_parquet_row_groups(files, columns=cols), sgd_kwargs={"alpha": 1e-5, "random_state": 7})

    # 3) Score each file and write to Parquet (scores only)
    for path in files:
        df = pl.read_parquet(path, columns=cols)
        df_w = woebin_ply(df, bins, keep_bins=False)
        scores = scorecard_ply(df_w, sc_online.points_map)
        out = pl.DataFrame({"score": scores})
        base = os.path.basename(path).replace(".parquet", "")
        out.write_parquet(os.path.join(OUTPUT_DIR, f"{base}_scores.parquet"))

    # Optional: Evaluate on last file (if y present) and save plots
    pred = sc_online.model.predict_proba(df_w.select([c for c in df_w.columns if c.endswith("_woe")]).to_numpy())[:, 1]
    perf = perf_eva(df_w[TARGET], pred, plot='both', save_prefix=os.path.join(OUTPUT_DIR, 'eval_last'))
    print("Evaluation on last file:", perf)

    # 4) Alternative: pure lazy streaming end-to-end (single output)
    lf = scan_parquet_select(files, columns=cols)
    lf_w = woebin_ply_lazy(lf, bins, keep_bins=False)
    lf_s = score_lazy(lf_w, sc_online.points_map, score_col='score', prefer_woe=True).select(['score'])
    sink_parquet_safe(lf_s, os.path.join(OUTPUT_DIR, 'scores_all.parquet'))


if __name__ == "__main__":
    main()
