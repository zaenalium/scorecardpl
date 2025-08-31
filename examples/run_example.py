import polars as pl
from scorecardpl import (
    split_df, var_filter, woebin, woebin_ply,
    scorecard, scorecard_ply, perf_eva,
    iv_summary, var_filter_by_iv, psi,
    bins_export_json, bins_import_json,
)


def make_data(n: int = 1000, seed: int = 42) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    # true probability via logistic on synthetic WOE-like effects
    logit = -3.0 + 0.04 * (age - 40) + 0.3 * (inc - 6) + (job == "C") * 0.5
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pl.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def main():
    df = make_data()
    train, valid = split_df(df, y="y", test_size=0.3, random_state=1)
    xs = [c for c in df.columns if c != "y"]
    train = var_filter(train, y="y", x=xs)

    # Advanced binning: chi2 for numeric + supervised categorical merge
    bins = woebin(
        train,
        y="y", x=xs,
        bins=6,
        method="chi2",
        chi2_params={"init_bins": 40},
        monotonic="auto",
        cat_max_bins=3,
        cat_method="target_rate",
    )
    print("IV summary:\n", iv_summary(bins))
    keep = var_filter_by_iv(bins, min_iv=0.02)
    print("Keep by IV:", keep)

    train_w = woebin_ply(train, bins)
    valid_w = woebin_ply(valid, bins)

    sc = scorecard(bins, y="y", data=train_w)
    X_valid = valid_w.select([c for c in valid_w.columns if c.endswith("_woe")]).to_numpy()
    pred = sc.model.predict_proba(X_valid)[:, 1]
    # Set plot to 'roc' or 'none' for headless environments
    perf = perf_eva(valid_w["y"], pred, plot='roc')
    print("Performance:", perf)

    scores = scorecard_ply(valid_w, sc.points_map)
    print("Scores head:", scores.head())

    # PSI between train and valid
    psi_df = psi(train, valid, bins)
    print("PSI:\n", psi_df)

    # Export/import bins
    bins_export_json(bins, "bins.json")
    bins2 = bins_import_json("bins.json")
    assert set(bins2.keys()) == set(bins.keys())
    print("Export/import OK")


if __name__ == "__main__":
    main()
