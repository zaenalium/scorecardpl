# Large-Scale Guide (100M+ rows)

This guide shows options for scaling to very large datasets.

## Sampling for Edges

Use `edges_sample_frac` or `edges_sample_n` to compute numeric edges on a sample (WOE/IV still computed on the working data subset you pass in):

```python
bins = woebin(df, y='y', x=xs, bins=6, method='chi2', chi2_params={'init_bins': 60}, edges_sample_frac=0.02)
```

## Streaming Training

Train a logistic model with `SGDClassifier.partial_fit` over Parquet files/row-groups:

```python
from scorecardpl import iter_parquet_row_groups, scorecard_sgd

sc = scorecard_sgd(bins, y='y', frames=iter_parquet_row_groups(files, columns=["y"]+xs))
```

## Lazy Transform and Scoring

Use Polars LazyFrame to avoid materializing the entire dataset:

```python
from scorecardpl import scan_parquet_select, woebin_ply_lazy, score_lazy, sink_parquet_safe

lf = scan_parquet_select(files, columns=["y"]+xs)
lf_w = woebin_ply_lazy(lf, bins, keep_bins=False)
lf_s = score_lazy(lf_w, sc.points_map, score_col='score')
sink_parquet_safe(lf_s.select(['score']), 'scores.parquet')
```

## Tips

- Prefer Parquet and column selection.
- Filter down variables first (IV/PSI/domain logic).
- Train on a representative sample or via streaming.
- Use `keep_bins=False` in `woebin_ply` to reduce memory.
- Set `POLARS_MAX_THREADS` to tune CPU usage.

