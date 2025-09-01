from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from .utils import to_pl_df


@dataclass
class ScorecardModel:
    model: LogisticRegression
    points_map: Dict[str, pl.DataFrame]  # per-var mapping: bin -> points
    base_score: float
    pdo: float
    odds: float

    def summary(self) -> pl.DataFrame:
        """Summarize the scorecard: per-variable coefficient estimate and points ranges.

        Returns a Polars DataFrame with columns:
        - variable, coef, nbin, woe_min, woe_max, points_min, points_max, points_mean, points_std
        Includes a row for '__INTERCEPT__' with its points.
        """
        return scorecard_summary(self)


def _train_lr_woe(df_woe: pl.DataFrame, y: str) -> LogisticRegression:
    feature_cols = [c for c in df_woe.columns if c.endswith("_woe")]
    X = df_woe.select(feature_cols).to_numpy()
    yv = df_woe.select(y).to_numpy().ravel().astype(int)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, yv)
    return lr


def scorecard(bins: Dict[str, pl.DataFrame], y: str, data: Any,
              pdo: float = 20.0, base_score: float = 600.0, odds: float = 50.0) -> ScorecardModel:
    """Train LR on WOE features and build points mapping (Polars)."""
    data = to_pl_df(data)
    feature_cols = [f"{v}_woe" for v in bins.keys() if f"{v}_woe" in data.columns]
    if not feature_cols:
        raise ValueError("No WOE columns found in `data`. Run `woebin_ply` first.")

    model = _train_lr_woe(data.select([y] + feature_cols), y=y)

    factor = pdo / np.log(2)
    offset = base_score + factor * np.log(odds)
    coef = dict(zip(feature_cols, model.coef_.ravel()))
    intercept = float(model.intercept_[0])

    points_map: Dict[str, pl.DataFrame] = {}
    for var, bdf in bins.items():
        f = f"{var}_woe"
        if f not in coef:
            continue
        c = float(coef[f])
        pm = bdf.select([
            pl.lit(var).alias("variable"),
            pl.col("bin"),
            pl.col("woe"),
            (-factor * c * pl.col("woe")).alias("points"),
        ])
        points_map[var] = pm

    # Store intercept as a pseudo-entry
    bias_df = pl.DataFrame({
        "variable": ["__INTERCEPT__"],
        "bin": ["__BIAS__"],
        "points": [offset - factor * intercept],
    })
    points_map["__INTERCEPT__"] = bias_df

    return ScorecardModel(model=model, points_map=points_map, base_score=base_score, pdo=pdo, odds=odds)


def scorecard_ply(df: Any, points_map: Dict[str, pl.DataFrame], score_col: str = "score") -> pl.Series:
    """Apply scorecard points map by joining `<var>_bin` to points and summing (Polars)."""
    out = to_pl_df(df)
    total = pl.Series(name=score_col, values=np.zeros(out.height, dtype=float))
    # get bias/intercept points
    bias = 0.0
    if "__INTERCEPT__" in points_map:
        bias = float(points_map["__INTERCEPT__"].select("points").item())
        total = total + bias
    # try fast path using WOE columns: contribution = k * <var>_woe, where k is constant per var
    for var, pm in points_map.items():
        if var == "__INTERCEPT__":
            continue
        woe_col = f"{var}_woe"
        used_fast_path = False
        if woe_col in out.columns:
            # derive k from points/woe ratio (constant per variable)
            df_pm = pm.filter(pl.col("woe") != 0.0)
            if df_pm.height > 0:
                ratio_df = df_pm.select((pl.col("points") / pl.col("woe")).alias("ratio"))
                k = float(ratio_df.select(pl.col("ratio").median()).to_series().item())
                total = total + (out.select(woe_col).to_series().fill_null(0.0) * k)
                used_fast_path = True
        if not used_fast_path:
            # fallback: join on bin labels
            bin_col = f"{var}_bin"
            if bin_col not in out.columns:
                continue
            m = pm.rename({"bin": bin_col, "points": f"{var}_pts"})
            out = out.join(m.select([bin_col, f"{var}_pts"]), on=bin_col, how="left")
            pts = out.select(f"{var}_pts").to_series().fill_null(0.0)
            total = total + pts
            out = out.drop(f"{var}_pts")
    return total


def scorecard_components(
    df: Any,
    points_map: Dict[str, pl.DataFrame],
    *,
    include_intercept: bool = True,
    total_col: str = "score",
) -> pl.DataFrame:
    """Compute per-variable point contributions and total score.

    - df: DataFrame containing `<var>_bin` columns (use `woebin_ply(..., keep_bins=True)`).
    - points_map: mapping built by scorecard or SHAP scorecard
    - include_intercept: include an `intercept_points` column if present
    - total_col: name of total score column

    Returns a Polars DataFrame with columns `[<var>_points..., intercept_points?, total_col]`.
    """
    out = to_pl_df(df)
    comp_cols: List[str] = []
    # per-variable join on bin
    for var, pm in points_map.items():
        if var == "__INTERCEPT__":
            continue
        bin_col = f"{var}_bin"
        if bin_col not in out.columns:
            # cannot compute component without bin labels
            continue
        var_col = f"{var}_points"
        m = pm.rename({"bin": bin_col, "points": var_col}).select([bin_col, var_col])
        out = out.join(m, on=bin_col, how="left")
        out = out.with_columns(pl.col(var_col).fill_null(0.0))
        comp_cols.append(var_col)
    # intercept
    intercept_col = None
    if include_intercept and "__INTERCEPT__" in points_map:
        try:
            bias = float(points_map["__INTERCEPT__"].select("points").item())
        except Exception:
            bias = 0.0
        intercept_col = "intercept_points"
        out = out.with_columns(pl.lit(bias).alias(intercept_col))
    # total = sum of components (+ intercept)
    total_inputs = comp_cols + ([intercept_col] if intercept_col else [])
    if total_inputs:
        out = out.with_columns(pl.sum_horizontal([pl.col(c) for c in total_inputs]).alias(total_col))
        return out.select(total_inputs + [total_col])
    # no components found
    return pl.DataFrame({total_col: []})

def scorecard_summary(sc: "ScorecardModel") -> pl.DataFrame:
    """Create a per-variable summary for a trained scorecard model.

    - sc: ScorecardModel
    Returns a Polars DataFrame with per-variable stats and a row for the intercept.
    """
    factor = sc.pdo / np.log(2)
    rows = []
    intercept_points = 0.0
    if "__INTERCEPT__" in sc.points_map:
        try:
            intercept_points = float(sc.points_map["__INTERCEPT__"].select("points").item())
        except Exception:
            intercept_points = 0.0
    for var, pm in sc.points_map.items():
        if var == "__INTERCEPT__":
            continue
        # estimate coefficient from points mapping: points = -factor * coef * woe
        coef = np.nan
        try:
            ratio_df = pm.filter(pl.col("woe") != 0.0).select((pl.col("points") / pl.col("woe")).alias("ratio"))
            if ratio_df.height > 0:
                k = float(ratio_df.select(pl.col("ratio").median()).to_series().item())
                coef = -k / factor
        except Exception:
            coef = np.nan
        agg = pm.select([
            pl.len().alias("nbin"),
            pl.col("woe").min().alias("woe_min"),
            pl.col("woe").max().alias("woe_max"),
            pl.col("woe").mean().alias("woe_mean"),
            pl.col("woe").std().alias("woe_std"),
            pl.col("points").min().alias("points_min"),
            pl.col("points").max().alias("points_max"),
            pl.col("points").mean().alias("points_mean"),
            pl.col("points").std().alias("points_std"),
        ]).to_dicts()[0]
        agg.update({"variable": var, "coef": float(coef) if coef == coef else np.nan})
        rows.append(agg)
    out = pl.DataFrame(rows)[
        [
            "variable",
            "coef",
            "nbin",
            "woe_min",
            "woe_max",
            "points_min",
            "points_max",
            "points_mean",
            "points_std",
        ]
    ].sort("variable")
    # append intercept row
    bias_df = pl.DataFrame({
        "variable": ["__INTERCEPT__"],
        "coef": [np.nan],
        "nbin": [1],
        "woe_min": [np.nan],
        "woe_max": [np.nan],
        "points_min": [intercept_points],
        "points_max": [intercept_points],
        "points_mean": [intercept_points],
        "points_std": [0.0],
    })
    return pl.concat([out, bias_df], how="vertical")
