import polars as pl

from scorecardpl import woebin, woebin_ply, scorecard_shap, scorecard_ply, perf_eva


def make_data(n: int = 5000, seed: int = 42) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    logit = -3.0 + 0.04 * (age - 40) + 0.3 * (inc - 6) + (job == "C") * 0.5
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pl.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def get_estimator():
    # prefer LightGBM/XGBoost if installed; else fallback to RandomForest
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=7)
    except Exception:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=7, tree_method='hist')
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=300, random_state=7)


def main():
    df = make_data()
    xs = [c for c in df.columns if c != "y"]
    bins = woebin(df, y="y", x=xs, bins=6)
    df_w = woebin_ply(df, bins)

    est = get_estimator()
    sc = scorecard_shap(bins, y="y", data=df_w, estimator=est, shap_sample_n=2000)

    scores = scorecard_ply(df_w, sc.points_map)
    pred = scores.to_numpy()  # not probabilities, but correlated with risk; use estimator for probabilities if needed
    perf = perf_eva(df_w["y"], pred, plot='none')
    print("SHAP scorecard performance (proxy):", perf)


if __name__ == "__main__":
    main()

