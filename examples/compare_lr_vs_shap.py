import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from scorecardpl import (
    split_df, var_filter, woebin, woebin_ply,
    scorecard, scorecard_ply, perf_eva,
    scorecard_shap, fit_points_calibrator,
)


def make_data(n: int = 20000, seed: int = 42) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C", "D"], size=n, p=[0.45, 0.3, 0.2, 0.05])
    # mild nonlinearity
    logit = -2.8 + 0.05 * (age - 40) + 0.35 * (inc - 6) + (job == "C") * 0.6 - (job == "D") * 0.2
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pl.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def main():
    df = make_data()
    train, valid = split_df(df, y="y", test_size=0.3, random_state=1)
    xs = [c for c in df.columns if c != "y"]
    train = var_filter(train, y="y", x=xs)

    # Bin on train
    bins = woebin(train, y="y", x=xs, bins=6, method="chi2", chi2_params={"init_bins": 50}, cat_max_bins=4)
    train_w = woebin_ply(train, bins)
    valid_w = woebin_ply(valid, bins)

    # 1) Logistic regression scorecard
    sc_lr = scorecard(bins, y="y", data=train_w)
    proba_lr = sc_lr.model.predict_proba(valid_w.select([c for c in valid_w.columns if c.endswith("_woe")]).to_numpy())[:, 1]
    perf_lr = perf_eva(valid_w["y"], proba_lr, plot='none')
    print("LR AUC/KS:", perf_lr)

    # 2) SHAP-based scorecard with RandomForest
    est = RandomForestClassifier(n_estimators=300, random_state=7)
    sc_shap = scorecard_shap(bins, y="y", data=train_w, estimator=est, shap_sample_n=5000)
    # raw points
    points_train = scorecard_ply(train_w, sc_shap.points_map)
    points_valid = scorecard_ply(valid_w, sc_shap.points_map)
    # calibrate points -> proba on train, evaluate on valid
    calibrator = fit_points_calibrator(train_w["y"].to_numpy(), points_train.to_numpy())
    proba_shap = calibrator.points_to_proba(points_valid.to_numpy())
    perf_shap = perf_eva(valid_w["y"], proba_shap, plot='none')
    print("SHAP AUC/KS:", perf_shap)


if __name__ == "__main__":
    main()

