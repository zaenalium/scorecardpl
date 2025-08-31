from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Literal, Mapping, Tuple

import numpy as np
import polars as pl
import os

from .utils import ensure_binary_target, to_pl_df, cut_expr


def _woe_iv_for_bin(df: pl.DataFrame, y: str, bin_col: str) -> pl.DataFrame:
    """Aggregate WOE/IV stats for a given bin column.

    Assumes binary target `y` in {0,1}. Null bins are kept as a separate row.
    """
    total_good = df.select((pl.col(y) == 0).sum()).item()
    total_bad = df.select((pl.col(y) == 1).sum()).item()
    eps = 1e-9
    base = (
        df.group_by(bin_col)
        .agg(
            count=pl.len(),
            bad=pl.col(y).sum(),
        )
        .with_columns(
            good=pl.col("count") - pl.col("bad"),
        )
    )
    agg = base.with_columns(
        bad_rate=(pl.col("bad") / max(total_bad, eps)),
        good_rate=(pl.col("good") / max(total_good, eps)),
    )
    agg = agg.with_columns(
        woe=((pl.col("good_rate") + eps) / (pl.col("bad_rate") + eps)).log(),
    )
    agg = agg.with_columns(
        iv=(pl.col("good_rate") - pl.col("bad_rate")) * pl.col("woe"),
    )
    return agg


def _maybe_sample_for_edges(
    s: pl.Series,
    y: Optional[pl.Series] = None,
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[pl.Series, Optional[pl.Series]]:
    if sample_n is None and (sample_frac is None or sample_frac <= 0 or sample_frac >= 1):
        return s, y
    df = pl.DataFrame({"__s__": s}) if y is None else pl.DataFrame({"__s__": s, "__y__": y})
    if sample_n is not None:
        sample_n = min(sample_n, df.height)
        sdf = df.sample(n=sample_n, shuffle=True, seed=seed)
    else:
        sdf = df.sample(fraction=float(sample_frac), shuffle=True, seed=seed)
    s_out = sdf["__s__"]
    y_out = sdf["__y__"] if y is not None else None
    return s_out, y_out


def _numeric_edges_quantile(s: pl.Series, bins: int) -> List[float]:
    """Compute numeric bin edges by quantiles, including [-inf, inf]."""
    vals = s.drop_nulls().to_numpy()
    if vals.size == 0:
        return [-np.inf, np.inf]
    q = max(1, min(bins, len(np.unique(vals))))
    probs = np.linspace(0, 1, q + 1)
    qs = np.quantile(vals, probs, method="nearest")
    edges = np.unique(qs)
    if edges.size < 2:
        edges = np.array([vals.min(), vals.max()])
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf])) if edges.size > 2 else np.array([-np.inf, np.inf])
    return edges.tolist()


def _numeric_edges_equal_width(s: pl.Series, bins: int) -> List[float]:
    """Compute equal-width numeric bin edges, including [-inf, inf]."""
    vals = s.drop_nulls().to_numpy()
    if vals.size == 0:
        return [-np.inf, np.inf]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo == hi:
        return [-np.inf, np.inf]
    inner = np.linspace(lo, hi, num=max(2, bins+1))[1:-1]
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    return edges.tolist()


def _numeric_edges_tree(s: pl.Series, y: pl.Series, bins: int, min_leaf_frac: float = 0.05, random_state: Optional[int] = 42) -> List[float]:
    """Derive edges via a decision tree on (s, y)."""
    try:
        from sklearn.tree import DecisionTreeClassifier  # type: ignore
    except Exception as e:
        raise ImportError(
            "Tree-based binning requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e
    X = s.to_numpy().reshape(-1, 1)
    yv = y.to_numpy().astype(int)
    mask = ~np.isnan(X.ravel()) & ~np.isnan(yv)
    Xc = X[mask]
    yc = yv[mask]
    if Xc.size == 0:
        return [-np.inf, np.inf]
    min_samples_leaf = max(1, int(len(yc) * min_leaf_frac))
    clf = DecisionTreeClassifier(max_leaf_nodes=bins, min_samples_leaf=min_samples_leaf, random_state=random_state)
    clf.fit(Xc, yc)
    thresholds = clf.tree_.threshold
    thresh = sorted(set([t for t in thresholds if t != -2.0]))
    if not thresh:
        return [-np.inf, np.inf]
    edges = [-np.inf] + thresh + [np.inf]
    return edges


def _chi2_pair_stat(g1_bad, g1_good, g2_bad, g2_good, eps: float = 1e-9) -> float:
    # 2x2 chi-squared statistic for adjacent groups
    o = np.array([g1_bad, g1_good, g2_bad, g2_good], dtype=float)
    t1 = g1_bad + g1_good
    t2 = g2_bad + g2_good
    tb = g1_bad + g2_bad
    tg = g1_good + g2_good
    total = t1 + t2 + eps
    e1_bad = tb * t1 / total
    e1_good = tg * t1 / total
    e2_bad = tb * t2 / total
    e2_good = tg * t2 / total
    e = np.array([e1_bad, e1_good, e2_bad, e2_good]) + eps
    chi2 = np.sum((o - e) ** 2 / e)
    return float(chi2)


def _numeric_edges_chi2(s: pl.Series, y: pl.Series, max_bins: int, init_bins: int = 50, min_bin_size: int = 1) -> List[float]:
    """Chi-squared style adjacent-bin merging to find edges.

    Starts from many quantile bins, then iteratively merges adjacent bins with
    the smallest chi2 statistic until reaching `max_bins`.
    """
    # prepare data
    x = s.to_numpy()
    yv = y.to_numpy().astype(float)
    mask = ~(np.isnan(x) | np.isnan(yv))
    x = x[mask]
    yv = yv[mask]
    if x.size == 0:
        return [-np.inf, np.inf]
    # initial edges via quantiles
    qbins = max(max_bins, min(init_bins, len(np.unique(x)) - 1)) if len(np.unique(x)) > 1 else 1
    edges = _numeric_edges_quantile(pl.Series("x", x), qbins)
    # digitize
    idx = np.digitize(x, edges[1:-1], right=True)
    # build counts per bin
    n_bins = len(edges) - 1
    bad = np.bincount(idx, weights=yv, minlength=n_bins).astype(float)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    good = cnt - bad
    # remove empty bins by merging with neighbors
    bins_list = [
        {
            "bad": float(bad[i]),
            "good": float(good[i]),
            "left": edges[i],
            "right": edges[i+1],
        }
        for i in range(n_bins)
        if cnt[i] >= min_bin_size
    ]
    if not bins_list:
        return [-np.inf, np.inf]
    # iterative merge until reaching max_bins
    def compute_pair_stats(blist):
        stats = []
        for i in range(len(blist) - 1):
            a, b = blist[i], blist[i+1]
            chi2 = _chi2_pair_stat(a["bad"], a["good"], b["bad"], b["good"])
            stats.append(chi2)
        return stats

    while len(bins_list) > max_bins:
        stats = compute_pair_stats(bins_list)
        if not stats:
            break
        j = int(np.argmin(stats))
        # merge j and j+1
        a = bins_list[j]
        b = bins_list[j+1]
        merged = {
            "bad": a["bad"] + b["bad"],
            "good": a["good"] + b["good"],
            "left": a["left"],
            "right": b["right"],
        }
        bins_list[j:j+2] = [merged]
    # finalize edges (preserve order and drop duplicates)
    out_edges = [-np.inf]
    for b in bins_list[:-1]:
        out_edges.append(b["right"])  # boundary between bins
    out_edges.append(np.inf)
    seen = set()
    out_edges = [x for x in out_edges if (x not in seen and not seen.add(x))]
    # guarantee -inf..inf
    if out_edges[0] != -np.inf:
        out_edges = [-np.inf] + out_edges
    if out_edges[-1] != np.inf:
        out_edges = out_edges + [np.inf]
    return out_edges


def _numeric_edges_isotonic(
    s: pl.Series,
    y: pl.Series,
    max_bins: int,
    increasing: Optional[bool] = None,
    tol: float = 1e-8,
) -> List[float]:
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
    except Exception as e:
        raise ImportError(
            "Isotonic binning requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e
    x = s.to_numpy()
    yv = y.to_numpy().astype(float)
    mask = ~(np.isnan(x) | np.isnan(yv))
    x = x[mask]
    yv = yv[mask]
    if x.size == 0:
        return [-np.inf, np.inf]
    # aggregate by unique x to stabilize
    order = np.argsort(x)
    xu = x[order]
    yu = yv[order]
    # group consecutive equal x
    uniq_vals, idx_start = np.unique(xu, return_index=True)
    idx_end = np.r_[idx_start[1:], len(xu)]
    means = np.array([yu[i:j].mean() for i, j in zip(idx_start, idx_end)])
    counts = np.array([j - i for i, j in zip(idx_start, idx_end)])
    # choose monotonic direction if not provided
    if increasing is None:
        corr = np.corrcoef(uniq_vals, means)[0, 1] if uniq_vals.size > 1 else 0.0
        inc_flag = bool(corr >= 0)
    else:
        inc_flag = bool(increasing)
    iso = IsotonicRegression(increasing=inc_flag, out_of_bounds="clip")
    yhat = iso.fit_transform(uniq_vals, means, sample_weight=counts)
    # find segments of constant yhat
    segs = []  # (start_idx, end_idx)
    start = 0
    for i in range(1, len(yhat)):
        if abs(yhat[i] - yhat[i-1]) > tol:
            segs.append((start, i))
            start = i
    segs.append((start, len(yhat)))
    # If too many segments, merge by smallest delta between adjacent segment means
    def seg_mean(seg):
        i, j = seg
        return float(yhat[i:j].mean())
    while len(segs) > max_bins:
        # compute deltas as |mean(next) - mean(current)|
        deltas = [abs(seg_mean(segs[k+1]) - seg_mean(segs[k])) for k in range(len(segs)-1)]
        kmin = int(np.argmin(deltas))
        new_seg = (segs[kmin][0], segs[kmin+1][1])
        segs[kmin:kmin+2] = [new_seg]
    # thresholds as right boundary value of each segment except last
    thresholds = [float(uniq_vals[j-1]) for (_, j) in segs[:-1]]
    out_edges = [-np.inf] + sorted(list(set(thresholds))) + [np.inf]
    return out_edges


def _enforce_monotonic_woe(stat: pl.DataFrame, direction: Optional[Literal['increasing','decreasing','auto']] = 'auto', min_bins: int = 3) -> pl.DataFrame:
    """Enforce monotonic WOE by merging adjacent bins with smallest deltas."""
    if 'ord' not in stat.columns:
        return stat
    d = stat.sort('ord')
    woe = d.select('woe').to_series().to_list()
    cnt = d.select('count').to_series().to_list()
    bad = d.select('bad').to_series().to_list()
    lower = d.select('lower').to_series().to_list() if 'lower' in d.columns else [None] * d.height
    upper = d.select('upper').to_series().to_list() if 'upper' in d.columns else [None] * d.height
    var = d.select('variable').to_series().to_list()[0] if d.height > 0 else None
    total_bad = float(np.sum(bad))
    total_good = float(np.sum(cnt) - total_bad)
    if d.height <= min_bins:
        return d
    # choose direction
    if direction == 'auto':
        # correlation between index and woe
        idx = np.arange(len(woe))
        try:
            corr = float(np.corrcoef(idx, np.array(woe))[0,1])
        except Exception:
            corr = 0.0
        inc = corr >= 0
    elif direction == 'increasing':
        inc = True
    elif direction == 'decreasing':
        inc = False
    else:
        inc = True

    def is_monotone(arr, inc_flag: bool) -> bool:
        if len(arr) <= 1:
            return True
        if inc_flag:
            return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
        return all(arr[i] >= arr[i+1] for i in range(len(arr)-1))

    # iterative merge based on smallest adjacent woe difference until monotone
    while (not is_monotone(woe, inc)) and (len(woe) > min_bins):
        deltas = [abs(woe[i+1] - woe[i]) for i in range(len(woe)-1)]
        j = int(np.argmin(deltas))
        # merge j and j+1
        cnt[j] = float(cnt[j] + cnt[j+1])
        bad[j] = float(bad[j] + bad[j+1])
        lower[j] = lower[j] if lower[j] is not None else lower[j+1]
        upper[j] = upper[j+1]
        # remove j+1
        for arr in (woe, cnt, bad, lower, upper):
            del arr[j+1]
        # recompute woe for j
        good_j = cnt[j] - bad[j]
        bad_rate = bad[j] / max(total_bad, 1e-9)
        good_rate = good_j / max(total_good, 1e-9)
        woe[j] = float(np.log((good_rate + 1e-9) / (bad_rate + 1e-9)))

    # rebuild DataFrame
    bins_labels = [f"({lower[i]}, {upper[i]}]" for i in range(len(woe))]
    good = [cnt[i] - bad[i] for i in range(len(woe))]
    bad_rate = [bad[i] / max(total_bad, 1e-9) for i in range(len(woe))]
    good_rate = [good[i] / max(total_good, 1e-9) for i in range(len(woe))]
    iv = [(good_rate[i] - bad_rate[i]) * woe[i] for i in range(len(woe))]
    n = len(woe)
    out = pl.DataFrame({
        'variable': pl.Series('variable', [var]*n, dtype=pl.Utf8),
        'bin': pl.Series('bin', bins_labels, dtype=pl.Utf8),
        'ord': pl.Series('ord', list(range(n)), dtype=pl.Int64),
        'lower': pl.Series('lower', lower, dtype=pl.Float64),
        'upper': pl.Series('upper', upper, dtype=pl.Float64),
        'count': pl.Series('count', cnt, dtype=pl.Float64),
        'bad': pl.Series('bad', bad, dtype=pl.Float64),
        'good': pl.Series('good', good, dtype=pl.Float64),
        'bad_rate': pl.Series('bad_rate', bad_rate, dtype=pl.Float64),
        'good_rate': pl.Series('good_rate', good_rate, dtype=pl.Float64),
        'woe': pl.Series('woe', woe, dtype=pl.Float64),
        'iv': pl.Series('iv', iv, dtype=pl.Float64),
    })
    return out


def _merge_categorical_stat(stat: pl.DataFrame, max_bins: int, method: Literal['target_rate','chi2','frequency'] = 'target_rate') -> pl.DataFrame:
    if stat.height <= max_bins:
        # add levels column with singletons
        return stat.with_columns(levels=pl.col('bin').map_elements(lambda x: [x], return_dtype=pl.List(pl.Utf8))).with_columns(ord=pl.int_range(0, pl.len()))
    d = stat.select(['bin','count','bad','good','bad_rate','good_rate'])
    # initialize groups sorted by bad_rate
    rows = d.sort('bad_rate').to_dicts()
    groups = [
        {
            'bin': r['bin'],
            'levels': [r['bin']],
            'count': float(r['count']),
            'bad': float(r['bad']),
            'good': float(r['good']),
            'bad_rate': float(r['bad_rate']),
            'good_rate': float(r['good_rate']),
        }
        for r in rows
    ]
    total_bad = float(sum(g['bad'] for g in groups))
    total_good = float(sum(g['good'] for g in groups))

    def recompute(g):
        g['count'] = g['bad'] + g['good']
        g['bad_rate'] = g['bad'] / max(total_bad, 1e-9)
        g['good_rate'] = g['good'] / max(total_good, 1e-9)
        gr = (g['good_rate'] + 1e-9) / (g['bad_rate'] + 1e-9)
        g['woe'] = float(np.log(gr))
        g['iv'] = (g['good_rate'] - g['bad_rate']) * g['woe']

    # iterative merge
    while len(groups) > max_bins:
        # compute pair metric over adjacent sorted by current bad_rate
        if method == 'chi2':
            scores = [
                _chi2_pair_stat(groups[i]['bad'], groups[i]['good'], groups[i+1]['bad'], groups[i+1]['good'])
                for i in range(len(groups)-1)
            ]
        elif method == 'frequency':
            scores = [
                min(groups[i]['count'], groups[i+1]['count'])
                for i in range(len(groups)-1)
            ]
        else:  # target_rate
            scores = [abs(groups[i]['bad']/max(groups[i]['count'],1e-9) - groups[i+1]['bad']/max(groups[i+1]['count'],1e-9)) for i in range(len(groups)-1)]
        j = int(np.argmin(scores))
        a, b = groups[j], groups[j+1]
        merged = {
            'bin': '|'.join(a['levels'] + b['levels']),
            'levels': a['levels'] + b['levels'],
            'bad': a['bad'] + b['bad'],
            'good': a['good'] + b['good'],
        }
        recompute(merged)
        groups[j:j+2] = [merged]
        # re-sort by bad rate to maintain adjacency meaning
        groups = sorted(groups, key=lambda g: g['bad']/max(g['count'],1e-9))

    # build DataFrame
    for g in groups:
        recompute(g)
    varname = stat.select('variable').to_series().to_list()[0]
    out = pl.DataFrame({
        'variable': [varname]*len(groups),
        'bin': [g['bin'] for g in groups],
        'count': [g['count'] for g in groups],
        'bad': [g['bad'] for g in groups],
        'good': [g['good'] for g in groups],
        'bad_rate': [g['bad_rate'] for g in groups],
        'good_rate': [g['good_rate'] for g in groups],
        'woe': [g.get('woe', 0.0) for g in groups],
        'iv': [g.get('iv', 0.0) for g in groups],
        'levels': [g['levels'] for g in groups],
    }).with_columns(ord=pl.int_range(0, pl.len()))
    return out


def woebin(
    df,
    y: str,
    x: Optional[Sequence[str]] = None,
    bins: int = 10,
    min_count: int = 10,
    method: Literal['quantile','equal_width','tree','custom','chi2','isotonic'] = 'quantile',
    breaks: Optional[Mapping[str, Sequence[float]]] = None,
    tree_params: Optional[Mapping[str, float]] = None,
    chi2_params: Optional[Mapping[str, float]] = None,
    isotonic_params: Optional[Mapping[str, float]] = None,
    # monotonic woe enforcement for numeric bins
    monotonic: Optional[Literal['auto','increasing','decreasing']] = None,
    monotonic_min_bins: int = 3,
    # categorical merging settings
    cat_max_bins: Optional[int] = None,
    cat_method: Literal['target_rate','chi2','frequency'] = 'target_rate',
    # sampling for numeric edge discovery (helps on very large data)
    edges_sample_frac: Optional[float] = None,
    edges_sample_n: Optional[int] = None,
    edges_sample_seed: Optional[int] = 42,
) -> Dict[str, pl.DataFrame]:
    """Create WOE/IV bins per variable (Polars).

    Returns a dict of Polars DataFrames per variable with columns:
    - variable, bin, count, bad, good, bad_rate, good_rate, woe, iv
    For numeric variables, also includes: lower, upper, ord
    """
    df = to_pl_df(df)
    if x is None:
        x = [c for c in df.columns if c != y]
    _ = ensure_binary_target(df.get_column(y))

    out: Dict[str, pl.DataFrame] = {}
    base = df.with_columns(pl.col(y).cast(pl.Int64))
    def _is_numeric_dtype(dt) -> bool:
        # Prefer modern Polars dtype API
        try:
            is_num = getattr(dt, "is_numeric", None)
            if callable(is_num):
                return bool(is_num())
            if isinstance(is_num, bool):
                return is_num
        except Exception:
            pass
        try:
            from polars import (
                Int8, Int16, Int32, Int64,
                UInt8, UInt16, UInt32, UInt64,
                Float32, Float64,
            )
            return dt in {Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64}
        except Exception:
            return False

    for col in x:
        s = base.get_column(col)
        if _is_numeric_dtype(s.dtype):
            # numeric: choose method and build edges
            if breaks and col in breaks:
                e = list(breaks[col])
                if len(e) == 0:
                    edges = [-np.inf, np.inf]
                else:
                    # ensure sorted and unique while preserving order
                    seen = set()
                    e = [v for v in e if (v not in seen and not seen.add(v))]
                    if any(isinstance(v, (float, int)) for v in e):
                        e = sorted(e)
                    # include boundaries
                    if e[0] != -np.inf:
                        e = [-np.inf] + e
                    if e[-1] != np.inf:
                        e = e + [np.inf]
                    if len(e) < 2:
                        e = [-np.inf, np.inf]
                    edges = e
            elif method == 'equal_width':
                s_samp, _ = _maybe_sample_for_edges(s, None, edges_sample_n, edges_sample_frac, edges_sample_seed)
                edges = _numeric_edges_equal_width(s_samp, bins)
            elif method == 'tree':
                tp = dict(tree_params or {})
                min_leaf_frac = float(tp.get('min_leaf_frac', 0.05))
                rs = tp.get('random_state', 42)
                s_samp, y_samp = _maybe_sample_for_edges(s, base.get_column(y), edges_sample_n, edges_sample_frac, edges_sample_seed)
                edges = _numeric_edges_tree(s_samp, y_samp, bins=bins, min_leaf_frac=min_leaf_frac, random_state=rs) 
            elif method == 'chi2':
                cp = dict(chi2_params or {})
                init_bins = int(cp.get('init_bins', max(20, bins*2)))
                min_bin_size = int(cp.get('min_bin_size', 1))
                s_samp, y_samp = _maybe_sample_for_edges(s, base.get_column(y), edges_sample_n, edges_sample_frac, edges_sample_seed)
                edges = _numeric_edges_chi2(s_samp, y_samp, max_bins=bins, init_bins=init_bins, min_bin_size=min_bin_size)
            elif method == 'isotonic':
                ip = dict(isotonic_params or {})
                increasing = ip.get('increasing', None)
                tol = float(ip.get('tol', 1e-8))
                s_samp, y_samp = _maybe_sample_for_edges(s, base.get_column(y), edges_sample_n, edges_sample_frac, edges_sample_seed)
                edges = _numeric_edges_isotonic(s_samp, y_samp, max_bins=bins, increasing=increasing, tol=tol)
            else:
                # default quantile
                s_samp, _ = _maybe_sample_for_edges(s, None, edges_sample_n, edges_sample_frac, edges_sample_seed)
                edges = _numeric_edges_quantile(s_samp, bins)
            labels = [f"({edges[i]}, {edges[i+1]}]" for i in range(len(edges)-1)]
            sub = base.select([pl.col(y), pl.col(col)])
            tmp = sub.with_columns(_bin=cut_expr(pl.col(col), edges, labels))
            stat = _woe_iv_for_bin(tmp.select([y, "_bin"]), y=y, bin_col="_bin")
            # reconstruct order and bounds
            bounds = pl.DataFrame({
                "bin": labels,
                "lower": edges[:-1],
                "upper": edges[1:],
                "ord": list(range(len(labels)))
            })
            stat = (
                stat.rename({"_bin": "bin"})
                .join(bounds, on="bin", how="left")
                .with_columns(variable=pl.lit(col))
                .select(["variable", "bin", "ord", "lower", "upper", "count", "bad", "good", "bad_rate", "good_rate", "woe", "iv"])
                .sort("ord")
            )
            if monotonic is not None:
                stat = _enforce_monotonic_woe(stat, direction=monotonic, min_bins=monotonic_min_bins)
        else:
            # categorical: group rare levels into __OTHER__
            sub = base.select([pl.col(y), pl.col(col)])
            vc_df = sub.drop_nulls(subset=[col]).group_by(col).len().rename({"len": "counts"})
            keep = set(vc_df.filter(pl.col("counts") >= min_count).select(col).to_series().to_list()) if vc_df.height > 0 else set()
            tmp = sub.with_columns(
                _bin=pl.when(pl.col(col).is_null())
                        .then(pl.lit("__NA__"))
                        .when(pl.col(col).cast(pl.Utf8).is_in(list(keep)))
                        .then(pl.col(col).cast(pl.Utf8))
                        .otherwise(pl.lit("__OTHER__"))
                        .alias("_bin")
            )
            stat = _woe_iv_for_bin(tmp.select([y, "_bin"]), y=y, bin_col="_bin")
            stat = (
                stat.rename({"_bin": "bin"})
                .with_columns(variable=pl.lit(col))
                .select(["variable", "bin", "count", "bad", "good", "bad_rate", "good_rate", "woe", "iv"])
            )
            # supervised merging for categories if requested
            if cat_max_bins is not None and cat_max_bins > 0:
                stat = _merge_categorical_stat(stat, max_bins=cat_max_bins, method=cat_method)
            else:
                # still add levels and ord for uniform downstream usage
                stat = stat.with_columns(levels=pl.col('bin').map_elements(lambda x: [x], return_dtype=pl.List(pl.Utf8)))
                stat = stat.with_columns(ord=pl.int_range(0, pl.len()))
        out[col] = stat
    return out


def woebin_plot(
    bins: Dict[str, pl.DataFrame],
    var: Optional[str] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """Plot WOE per bin for a variable or all variables (Polars).

    - save_dir: if provided, saves each plot as `<save_dir>/woe_<var>.png`.
    - show: whether to display the plot windows (ignored if running headless).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install matplotlib`."
        ) from e
    vars_to_plot = [var] if var else list(bins.keys())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    saved: Dict[str, Optional[str]] = {}
    for v in vars_to_plot:
        d = bins[v]
        # extract columns as Python lists
        bin_labels = d.select("bin").to_series().to_list()
        woe_vals = d.select("woe").to_series().to_list()
        plt.figure(figsize=(6, 3))
        plt.title(f"WOE - {v}")
        plt.bar(range(len(woe_vals)), woe_vals, tick_label=bin_labels, alpha=0.8)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('WOE')
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f"woe_{v}.png")
            plt.savefig(path, dpi=144, bbox_inches='tight')
            if not show:
                plt.close()
            saved[v] = path
        if show and not save_dir:
            plt.show()
            saved[v] = None
    return saved if save_dir or show else {}
