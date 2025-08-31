from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from .transform import woebin_ply
from .scorecard import scorecard_ply
from .utils import to_pl_df


@dataclass
class PointsCalibrator:
    method: str
    intercept_: float = 0.0
    slope_: float = 1.0
    iso_: Optional[IsotonicRegression] = None

    def points_to_proba(self, points: Any) -> np.ndarray:
        vals = np.asarray(points, dtype=float).ravel()
        if self.method == "isotonic":
            if self.iso_ is None:
                raise RuntimeError("Isotonic calibrator not fitted")
            return self.iso_.predict(vals)
        # logistic
        z = self.intercept_ + self.slope_ * vals
        return 1.0 / (1.0 + np.exp(-z))

    def proba_to_points(self, proba: Any) -> np.ndarray:
        if self.method == "isotonic":
            raise NotImplementedError("Inverse mapping is not defined for isotonic calibration")
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
        logit = np.log(p / (1 - p))
        return (logit - self.intercept_) / max(self.slope_, 1e-12)


def fit_points_calibrator(y_true: Any, points: Any, method: str = "logistic") -> PointsCalibrator:
    """Fit a 1D calibration mapping from points to P(y=1).

    - method: 'logistic' (default) or 'isotonic'
    """
    y = np.asarray(y_true, dtype=int).ravel()
    X = np.asarray(points, dtype=float).ravel()
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X, y)
        return PointsCalibrator(method="isotonic", iso_=iso)
    # logistic
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X.reshape(-1, 1), y)
    a = float(lr.intercept_[0])
    b = float(lr.coef_.ravel()[0])
    return PointsCalibrator(method="logistic", intercept_=a, slope_=b)


def calibrate_scorecard_from_data(
    bins: dict,
    points_map: dict,
    y: str,
    data: Any,
) -> PointsCalibrator:
    """Convenience: transform data, compute points, and fit calibration.

    Works for both LR-based and SHAP-based points maps. For SHAP, ensure `woebin_ply`
    keeps `<var>_bin` columns (default) so `scorecard_ply` can join per-bin points.
    """
    df = to_pl_df(data)
    df_w = woebin_ply(df, bins)  # keep_bins=True
    pts = scorecard_ply(df_w, points_map).to_numpy()
    yv = df_w.select(y).to_numpy().ravel().astype(int)
    return fit_points_calibrator(yv, pts)


def make_points_proba_fn(calibrator: PointsCalibrator):
    """Return a callable `predict_proba(points)` using a fitted calibrator."""
    def _predict(points_any):
        return calibrator.points_to_proba(points_any)
    return _predict


def fit_scorecard_predictor(
    bins: dict,
    points_map: dict,
    y: str,
    data: Any,
    method: str = "logistic",
):
    """Fit a calibrator and return (calibrator, predict_proba_fn) for future data.

    The returned `predict_proba_fn(df)` will:
    - transform df via woebin_ply
    - compute scorecard points
    - map points to probabilities using the fitted calibrator
    """
    cal = calibrate_scorecard_from_data(bins, points_map, y=y, data=data)
    def _predict_df(df_any):
        d = to_pl_df(df_any)
        dw = woebin_ply(d, bins)
        pts = scorecard_ply(dw, points_map).to_numpy()
        return cal.points_to_proba(pts)
    return cal, _predict_df
