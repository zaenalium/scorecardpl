from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import polars as pl
from sklearn.linear_model import SGDClassifier

from .transform import woebin_ply


def iter_parquet(files: Sequence[str], columns: Optional[Sequence[str]] = None) -> Iterable[pl.DataFrame]:
    for path in files:
        df = pl.read_parquet(path, columns=list(columns) if columns else None)
        yield df


def iter_parquet_row_groups(files: Sequence[str], columns: Optional[Sequence[str]] = None) -> Iterable[pl.DataFrame]:
    """Yield Polars DataFrames per Parquet row group using PyArrow if available.

    Falls back to whole-file reads if PyArrow is not installed.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        # fallback
        yield from iter_parquet(files, columns)
        return
    cols = list(columns) if columns else None
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=cols)
            yield pl.from_arrow(table)


@dataclass
class ScorecardOnlineModel:
    model: Any
    points_map: Dict[str, pl.DataFrame]
    base_score: float
    pdo: float
    odds: float


def scorecard_sgd(
    bins: Dict[str, pl.DataFrame],
    y: str,
    frames: Iterable[Union[pl.DataFrame, Any]],
    *,
    classes: Sequence[int] = (0, 1),
    sgd_kwargs: Optional[Dict[str, Any]] = None,
    pdo: float = 20.0,
    base_score: float = 600.0,
    odds: float = 50.0,
) -> ScorecardOnlineModel:
    """Train a streaming logistic model (SGDClassifier) on WOE features.

    - bins: output of woebin
    - y: target column
    - frames: iterable yielding data batches (Polars DataFrame or pandas DataFrame)
    - classes: class labels for partial_fit
    - sgd_kwargs: passed to SGDClassifier (defaults set for log-loss)
    """
    feat_cols = [f"{v}_woe" for v in bins.keys()]
    kwargs = dict(loss="log_loss", max_iter=1, tol=None)
    if sgd_kwargs:
        kwargs.update(sgd_kwargs)
    clf = SGDClassifier(**kwargs)

    first = True
    for batch in frames:
        if not isinstance(batch, pl.DataFrame):
            batch = pl.from_pandas(batch)
        batch_w = woebin_ply(batch, bins, keep_bins=False)
        # ensure missing WOE columns are created (fill 0)
        for col in feat_cols:
            if col not in batch_w.columns:
                batch_w = batch_w.with_columns(pl.lit(0.0).alias(col))
        X = batch_w.select(feat_cols).to_numpy()
        yv = batch_w.select(y).to_numpy().ravel().astype(int)
        if first:
            clf.partial_fit(X, yv, classes=np.array(classes))
            first = False
        else:
            clf.partial_fit(X, yv)

    # Build points map like in standard scorecard
    factor = pdo / np.log(2)
    offset = base_score + factor * np.log(odds)
    coef = dict(zip(feat_cols, clf.coef_.ravel()))
    intercept = float(clf.intercept_[0])

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
    # intercept/bias
    points_map['__INTERCEPT__'] = pl.DataFrame({
        'variable': ['__INTERCEPT__'],
        'bin': ['__BIAS__'],
        'woe': [0.0],
        'points': [offset - factor * intercept],
    })

    return ScorecardOnlineModel(model=clf, points_map=points_map, base_score=base_score, pdo=pdo, odds=odds)
