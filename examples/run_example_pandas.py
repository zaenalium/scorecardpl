import numpy as np
import pandas as pd

from scorecardpl import (
    split_df, var_filter, woebin, woebin_ply,
    scorecard, scorecard_ply, perf_eva,
    iv_summary, var_filter_by_iv, psi,
    bins_export_json, bins_import_json,
)


def make_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    logit = -3.0 + 0.04 * (age - 40) + 0.3 * (inc - 6) + (job == "C") * 0.5
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def main():
    # Start with pandas input
    df = make_data()

    # Split (function returns Polars; convert back to pandas for a pandas-only flow)
    train_pl, valid_pl = split_df(df, y="y", test_size=0.3, random_state=1)
    train = train_pl.to_pandas()
    valid = valid_pl.to_pandas()

    xs = [c for c in df.columns if c != "y"]

    # Variable filtering (returns Polars) -> to pandas
    train = var_filter(train, y="y", x=xs).to_pandas()

    # Binning (accepts pandas, returns dict of Polars DataFrames)
    bins = woebin(train, y="y", x=xs, bins=5)
    print("IV summary:\n", iv_summary(bins).to_pandas())
    keep = var_filter_by_iv(bins, min_iv=0.02)
    print("Keep by IV:", keep)

    # Apply bins (returns Polars) -> to pandas
    train_w = woebin_ply(train, bins).to_pandas()
    valid_w = woebin_ply(valid, bins).to_pandas()

    # Train scorecard on WOE columns
    sc = scorecard(bins, y="y", data=train_w)
    woe_cols = [c for c in valid_w.columns if c.endswith("_woe")]
    X_valid = valid_w[woe_cols].values
    pred = sc.model.predict_proba(X_valid)[:, 1]
    perf = perf_eva(valid_w["y"].values, pred, plot='both')
    print("Performance:", perf)

    # Score using points map (returns Polars Series) -> to pandas Series
    scores = scorecard_ply(valid_w, sc.points_map).to_pandas()
    print("Scores head:\n", scores.head())

    # PSI between train and valid (accepts pandas inputs)
    psi_df = psi(train, valid, bins)
    print("PSI:\n", psi_df.to_pandas())

    # Export/import bins
    bins_export_json(bins, "bins.json")
    bins2 = bins_import_json("bins.json")
    assert set(bins2.keys()) == set(bins.keys())
    print("Export/import OK")


if __name__ == "__main__":
    main()

