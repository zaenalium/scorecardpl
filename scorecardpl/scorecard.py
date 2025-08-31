from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

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
        if woe_col in out.columns:
            # derive k from points/woe ratio (constant per variable)
            df_pm = pm.filter(pl.col("woe") != 0.0)
            if df_pm.height == 0:
                continue
            ratio_df = df_pm.select((pl.col("points") / pl.col("woe")).alias("ratio"))
            k = float(ratio_df.select(pl.col("ratio").median()).to_series().item())
            total = total + (out.select(woe_col).to_series().fill_null(0.0) * k)
        else:
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
