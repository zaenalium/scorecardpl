# Quickstart

```python
import polars as pl
from scorecardpl import split_df, var_filter, woebin, woebin_ply, scorecard, perf_eva, scorecard_ply

df = pl.read_csv('your_data.csv')  # or Parquet for speed
train, valid = split_df(df, y='y', test_size=0.3, random_state=42)
xs = [c for c in df.columns if c != 'y']
train = var_filter(train, y='y', x=xs)

bins = woebin(train, y='y', x=xs, bins=6, method='chi2', chi2_params={'init_bins': 60}, cat_max_bins=4)
train_w = woebin_ply(train, bins)
valid_w = woebin_ply(valid, bins)

sc = scorecard(bins, y='y', data=train_w)
pred = sc.model.predict_proba(valid_w.select([c for c in valid_w.columns if c.endswith('_woe')]).to_numpy())[:, 1]
perf = perf_eva(valid_w['y'], pred, plot='both', save_prefix='plots/quickstart')
print(perf)

scores = scorecard_ply(valid_w, sc.points_map)
print(scores.head())
```

