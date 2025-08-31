import polars as pl

from sklearn.ensemble import RandomForestClassifier

from scorecardpl import (
    split_df, var_filter, woebin, woebin_ply,
    scorecard_shap, scorecard_ply, perf_eva,
    fit_points_calibrator, make_points_proba_fn,
)


def make_data(n: int = 20000, seed: int = 42) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C", "D"], size=n, p=[0.45, 0.3, 0.2, 0.05])
    logit = -2.8 + 0.05 * (age - 40) + 0.35 * (inc - 6) + (job == "C") * 0.6 - (job == "D") * 0.2
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pl.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def main():
    df = make_data()
    train, valid = split_df(df, y="y", test_size=0.3, random_state=7)
    xs = [c for c in df.columns if c != "y"]
    train = var_filter(train, y="y", x=xs)

    bins = woebin(train, y="y", x=xs, bins=6, method="chi2", chi2_params={"init_bins": 50}, cat_max_bins=4)
    train_w = woebin_ply(train, bins)
    valid_w = woebin_ply(valid, bins)

    # Build SHAP-based points
    est = RandomForestClassifier(n_estimators=300, random_state=7)
    sc_shap = scorecard_shap(bins, y="y", data=train_w, estimator=est, shap_sample_n=5000)

    points_tr = scorecard_ply(train_w, sc_shap.points_map)
    points_va = scorecard_ply(valid_w, sc_shap.points_map)

    # Logistic calibration
    cal_log = fit_points_calibrator(train_w["y"], points_tr, method="logistic")
    pred_log = cal_log.points_to_proba(points_va)
    perf_log = perf_eva(valid_w["y"], pred_log, plot='none')
    print("Logistic-calibrated AUC/KS:", perf_log)

    # Isotonic calibration
    cal_iso = fit_points_calibrator(train_w["y"], points_tr, method="isotonic")
    pred_iso = cal_iso.points_to_proba(points_va)
    perf_iso = perf_eva(valid_w["y"], pred_iso, plot='none')
    print("Isotonic-calibrated AUC/KS:", perf_iso)

    # Turn calibrator into a predict_proba(points) function
    predict_proba_points = make_points_proba_fn(cal_log)
    _ = predict_proba_points(points_va)


if __name__ == "__main__":
    main()

