import os
import tempfile

import numpy as np
import polars as pl

from scorecardpl import (
    split_df,
    var_filter,
    woebin,
    woebin_ply,
    scorecard,
    scorecard_ply,
    perf_eva,
    iv_summary,
    var_filter_by_iv,
    psi,
    bins_export_json,
    bins_import_json,
    woebin_plot,
)
from scorecardpl.utils import cut_expr


def make_data(n: int = 500, seed: int = 123) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    logit = -2.5 + 0.04 * (age - 40) + 0.25 * (inc - 6) + (job == "C") * 0.5
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pl.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def test_cut_expr_semantics():
    df = pl.DataFrame({"x": [-5.0, 0.0, 1.0, 10.0, 11.0, None]})
    edges = [-np.inf, 0.0, 10.0, np.inf]
    labels = ["A", "B", "C"]
    got = df.with_columns(cut_expr(pl.col("x"), edges, labels).alias("bin")).select("bin").to_series().to_list()
    assert got == ["A", "A", "B", "B", "C", None]


def test_binning_transform_scorecard_pipeline(tmp_path):
    df = make_data()
    train, valid = split_df(df, y="y", test_size=0.3, random_state=1)
    xs = [c for c in df.columns if c != "y"]
    train = var_filter(train, y="y", x=xs)

    bins = woebin(
        train,
        y="y",
        x=xs,
        bins=5,
        method="chi2",
        chi2_params={"init_bins": 25},
        monotonic="auto",
        cat_max_bins=3,
    )
    # IV summary and selection
    ivsum = iv_summary(bins)
    assert set(ivsum.columns) == {"variable", "iv", "nbin"}
    keep = var_filter_by_iv(bins, min_iv=0.001)
    assert len(keep) > 0

    # Transform
    train_w = woebin_ply(train, bins)
    valid_w = woebin_ply(valid, bins)
    for v in xs:
        assert f"{v}_bin" in train_w.columns
        assert f"{v}_woe" in train_w.columns

    # Model + performance
    sc = scorecard(bins, y="y", data=train_w)
    X_valid = valid_w.select([c for c in valid_w.columns if c.endswith("_woe")]).to_numpy()
    pred = sc.model.predict_proba(X_valid)[:, 1]
    perf = perf_eva(valid_w["y"], pred, plot="none")
    assert set(perf.keys()) == {"auc", "ks"}

    # Points application
    scores = scorecard_ply(valid_w, sc.points_map)
    assert isinstance(scores, pl.Series)
    assert scores.len() == valid_w.height

    # PSI
    psi_df = psi(train, valid, bins)
    assert set(psi_df.columns) == {"variable", "psi"}
    assert psi_df.height > 0

    # Export/import bins
    json_path = tmp_path / "bins.json"
    bins_export_json(bins, str(json_path))
    bins2 = bins_import_json(str(json_path))
    assert set(bins2.keys()) == set(bins.keys())


def test_woebin_plot_saves(tmp_path):
    df = make_data(n=200, seed=7)
    xs = [c for c in df.columns if c != "y"]
    bins = woebin(df, y="y", x=xs, bins=4, method="quantile")
    outdir = tmp_path / "plots"
    woebin_plot(bins, var="age", save_dir=str(outdir), show=False)
    files = os.listdir(outdir)
    assert any(name.startswith("woe_age") and name.endswith(".png") for name in files)

