import os
import polars as pl

from scorecardpl import (
    split_df, var_filter, woebin, woebin_ply, woebin_plot,
    scorecard, perf_eva, iv_summary, write_method_report
)


def make_data(n: int = 2000, seed: int = 13) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, size=n)
    inc = rng.normal(6.0, 2.0, size=n)
    job = rng.choice(["A", "B", "C", "D"], size=n, p=[0.45, 0.3, 0.2, 0.05])
    # non-linear relation for variety
    logit = -2.8 + 0.05 * (age - 40) + 0.35 * (inc - 6) + (job == "C") * 0.6 - (job == "D") * 0.2
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pl.DataFrame({"y": y, "age": age, "inc": inc, "job": job})


def main():
    out_dir = "examples/plots"
    os.makedirs(out_dir, exist_ok=True)

    df = make_data()
    train, valid = split_df(df, y="y", test_size=0.3, random_state=7)
    xs = [c for c in df.columns if c != "y"]
    train = var_filter(train, y="y", x=xs)

    configs = {
        "quantile": dict(method="quantile"),
        "tree": dict(method="tree", tree_params={"min_leaf_frac": 0.03}),
        "chi2": dict(method="chi2", chi2_params={"init_bins": 50}),
        "isotonic": dict(method="isotonic"),
    }

    rows = []
    for name, kw in configs.items():
        bins = woebin(train, y="y", x=xs, bins=6, monotonic="auto", cat_max_bins=4, cat_method="target_rate", **kw)
        # Save WOE plots per variable
        woebin_plot(bins, save_dir=os.path.join(out_dir, name), show=False)
        iv_sum = iv_summary(bins).to_pandas()
        print(f"=== {name} ===\n", iv_sum)

        train_w = woebin_ply(train, bins)
        valid_w = woebin_ply(valid, bins)
        sc = scorecard(bins, y="y", data=train_w)
        X_valid = valid_w.select([c for c in valid_w.columns if c.endswith("_woe")]).to_numpy()
        pred = sc.model.predict_proba(X_valid)[:, 1]
        perf = perf_eva(valid_w["y"], pred, plot='both', save_prefix=os.path.join(out_dir, f"{name}"))
        rows.append({"method": name, **perf})

    res = pl.DataFrame(rows).sort("auc", descending=True)
    print("\nComparison (higher AUC, KS is better):\n", res)
    write_method_report(rows, os.path.join(out_dir, "methods_summary"))


if __name__ == "__main__":
    main()
