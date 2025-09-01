from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import polars as pl

from .utils import to_pl_df
from .transform import woebin_ply
from .scorecard import scorecard_ply, scorecard_components


@dataclass
class SHAPScorecardModel:
    estimator: Any
    points_map: Dict[str, pl.DataFrame]
    base_score: float
    pdo: float
    odds: float

    # Convenience inference helpers
    def predict_points(self, df: Any, bins: Dict[str, pl.DataFrame]) -> pl.Series:
        d = to_pl_df(df)
        dw = woebin_ply(d, bins)  # keep bins for join path
        return scorecard_ply(dw, self.points_map)

    def predict_proba(self, df: Any, bins: Dict[str, pl.DataFrame], calibrator: Optional[Any] = None) -> np.ndarray:
        d = to_pl_df(df)
        # estimator proba if no calibrator provided
        dw = woebin_ply(d, bins, keep_bins=False)
        feat_cols = [c for c in dw.columns if c.endswith("_woe")]
        X = dw.select(feat_cols).to_numpy() if feat_cols else np.zeros((dw.height, 0))
        if calibrator is None:
            try:
                return self.estimator.predict_proba(X)[:, 1]
            except Exception:
                # fallback to raw points scaled via logistic with default params (rough)
                pts = scorecard_ply(woebin_ply(d, bins), self.points_map).to_numpy()
                # center/scale heuristic
                m, s = np.median(pts), np.std(pts) + 1e-6
                z = (pts - m) / s
                return 1.0 / (1.0 + np.exp(-z))
        # calibrated probability from points
        pts = scorecard_ply(woebin_ply(d, bins), self.points_map).to_numpy()
        return calibrator.points_to_proba(pts)

    def predict_points_components(self, df: Any, bins: Dict[str, pl.DataFrame], total_col: str = "score") -> pl.DataFrame:
        """Return per-variable point contributions and total score for SHAP scorecard.

        Ensures `<var>_bin` columns are present via `woebin_ply` and joins to the
        SHAP-derived `points_map` for each variable.
        """
        d = to_pl_df(df)
        dw = woebin_ply(d, bins)  # keep_bins=True
        return scorecard_components(dw, self.points_map, include_intercept=True, total_col=total_col)


def _ensure_shap():
    try:
        import shap  # noqa: F401
    except Exception as e:
        raise ImportError("The 'shap' package is required for SHAP-based scorecard. Install via `pip install shap`.\nOriginal error: " + str(e))


def _compute_shap(estimator, X: np.ndarray, feature_names: Sequence[str], nsample: int = 20000):
    import shap

    n, nf = X.shape[0], len(feature_names)
    if nsample and n > nsample:
        idx = np.random.default_rng(42).choice(n, size=nsample, replace=False)
        Xs = X[idx]
        index = idx
    else:
        Xs = X
        index = np.arange(n)

    # Prefer TreeExplainer for tree models; fall back to generic Explainer.
    # For newer shap versions, TreeExplainer(model_output='log_odds') may not be supported
    # with default perturbation, so we try it first, then fall back to a generic
    # Explainer with a logit link to obtain contributions on the log-odds scale.
    try:
        explainer = shap.TreeExplainer(estimator, feature_names=feature_names, model_output="log_odds")
        exp = explainer(Xs)
    except Exception:
        try:
            explainer = shap.Explainer(estimator, feature_names=feature_names, link=getattr(shap.links, "logit", shap.links.identity))
            exp = explainer(Xs)
        except Exception as e:
            raise RuntimeError(f"Failed to create SHAP explainer: {e}")

    # Normalize output to (ns, nf) and base scalar.
    values = getattr(exp, "values", None)
    if values is None:
        values = exp
    values = np.array(values)

    # Handle shape differences across SHAP versions:
    # - Old: (ns, nclass, nf) => select last class => (ns, nf)
    # - New: (ns, nf, nclass) => select last class => (ns, nf)
    # - Binary simplified: (ns, nf)
    if values.ndim == 3:
        if values.shape[1] == nf:  # (ns, nf, nclass)
            values = values[:, :, -1]
        elif values.shape[2] == nf:  # (ns, nclass, nf)
            values = values[:, -1, :]
        else:
            # Best-effort: choose the axis that matches nf as features
            if nf in values.shape:
                ax = list(values.shape).index(nf)
                # Move features axis to position 1 and pick last class along the other axis
                values = np.moveaxis(values, ax, 1)
                # now values is (ns, nf, ?)
                values = values[:, :, -1]
            else:
                raise RuntimeError(f"Unexpected SHAP values shape {values.shape}; cannot align features")
    elif values.ndim == 2:
        if values.shape[1] != nf:
            # Some explainers may return transposed; attempt to fix
            if values.shape[0] == nf:
                values = values.T
            else:
                raise RuntimeError(f"SHAP values shape {values.shape} does not match n_features={nf}")

    base = getattr(exp, "base_values", 0.0)
    base = np.array(base)
    # For classification: base may be (ns, nclass); choose positive class
    if base.ndim == 2:
        base = base[:, -1]
    if base.ndim > 0:
        base = float(np.mean(base))
    else:
        base = float(base)
    return values, base, index


def scorecard_shap(
    bins: Dict[str, pl.DataFrame],
    y: str,
    data: Any,
    estimator: Any,
    *,
    shap_sample_n: int = 20000,
    pdo: float = 20.0,
    base_score: float = 600.0,
    odds: float = 50.0,
) -> SHAPScorecardModel:
    """Train any sklearn classifier on WOE features and build a scorecard via SHAP.

    - estimator: any scikit-learn-compatible binary classifier with .fit/.predict_proba
    - shap_sample_n: number of rows to sample for SHAP value estimation (speed)

    Returns a SHAPScorecardModel with a bin-level points map suitable for `scorecard_ply`.
    Note: For SHAP-derived points, keep `<var>_bin` columns when transforming, so
    `scorecard_ply` can join points per bin. The fast WOE-based path is skipped.
    """
    _ensure_shap()

    df = to_pl_df(data)
    dfw = woebin_ply(df, bins, )  # keep_bins=True by default (needed for SHAP mapping)
    feat_cols = [f"{v}_woe" for v in bins.keys() if f"{v}_woe" in dfw.columns]
    if not feat_cols:
        raise ValueError("No WOE columns found after transform.")

    X = dfw.select(feat_cols).to_numpy()
    yv = dfw.select(y).to_numpy().ravel().astype(int)
    estimator.fit(X, yv)

    values, base_logit, index = _compute_shap(estimator, X, feature_names=feat_cols, nsample=shap_sample_n)
    # Build points map per variable/bin using SHAP contributions
    factor = pdo / np.log(2)
    points_map: Dict[str, pl.DataFrame] = {}
    df_sample = dfw.slice(int(index.min()), len(index)) if len(index) == len(dfw) else dfw.take(index.tolist())

    for j, fcol in enumerate(feat_cols):
        var = fcol[:-4]  # strip _woe
        bin_col = f"{var}_bin"
        if bin_col not in df_sample.columns:
            # cannot build bin mapping without bin labels
            continue
        contrib = values[:, j]
        dd = pl.DataFrame({
            'bin': df_sample.select(bin_col).to_series().cast(pl.Utf8),
            'contrib': contrib,
        })
        stat = dd.group_by('bin').agg(contrib_mean=pl.col('contrib').mean()).sort('bin')
        # Convert SHAP log-odds contributions to points
        pm = stat.select([
            pl.lit(var).alias('variable'),
            pl.col('bin'),
            pl.lit(0.0).alias('woe'),  # ensure scorecard_ply skips WOE fast path
            (-factor * pl.col('contrib_mean')).alias('points'),
        ])
        points_map[var] = pm

    # Intercept from SHAP base value
    offset = base_score + factor * np.log(odds)
    bias_points = offset - factor * base_logit
    points_map['__INTERCEPT__'] = pl.DataFrame({
        'variable': ['__INTERCEPT__'],
        'bin': ['__BIAS__'],
        'woe': [0.0],
        'points': [bias_points],
    })

    # Ensure orientation: higher score for good (y=0), lower for bad (y=1)
    try:
        # Use the same sampled frame to estimate orientation
        if y in df_sample.columns:
            scores = scorecard_ply(df_sample, points_map).to_numpy()
            yv_s = df_sample.select(y).to_numpy().ravel().astype(int)
            mask0 = yv_s == 0
            mask1 = yv_s == 1
            if mask0.any() and mask1.any():
                mu0 = float(np.mean(scores[mask0]))
                mu1 = float(np.mean(scores[mask1]))
                if not np.isnan(mu0) and not np.isnan(mu1) and mu0 <= mu1:
                    # Flip sign of per-variable contributions; keep intercept unchanged
                    for k in list(points_map.keys()):
                        if k == '__INTERCEPT__':
                            continue
                        points_map[k] = points_map[k].with_columns((-pl.col('points')).alias('points'))
    except Exception:
        pass

    return SHAPScorecardModel(estimator=estimator, points_map=points_map, base_score=base_score, pdo=pdo, odds=odds)
