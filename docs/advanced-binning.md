# Advanced Binning

scorecardpl supports multiple supervised and unsupervised numeric binning methods and categorical merging strategies.

## Numeric Methods

- quantile: standard quantile edges.
- equal_width: evenly spaced edges between min and max.
- tree: supervised DecisionTree splits.
- chi2: chi-square adjacent merging from fine-grained quantiles.
- isotonic: monotonic supervised segmentation via Isotonic Regression.
- custom: user-supplied breaks per variable.

```python
bins = woebin(
    df, y='y', x=['age','inc'], bins=6, method='tree',
    tree_params={'min_leaf_frac': 0.03}, monotonic='auto'
)

bins = woebin(
    df, y='y', x=['age','inc'], bins=6, method='chi2',
    chi2_params={'init_bins': 60}, monotonic='auto'
)

bins = woebin(
    df, y='y', x=['age','inc'], bins=6, method='isotonic'
)
```

### Monotonic WOE Enforcement

Set `monotonic='auto'|'increasing'|'decreasing'` to merge adjacent bins until WOE is monotone.

## Categorical Merging

Control the number of categorical bins with `cat_max_bins` and choose a merging method:

- target_rate (default): merges adjacent groups with closest bad-rate.
- chi2: merges least different by chi-square.
- frequency: merges smallest-count neighbors.

```python
bins = woebin(df, y='y', x=['job'], cat_max_bins=4, cat_method='target_rate')
```

## Plotting WOE

```python
from scorecardpl import woebin_plot
# Returns a mapping of variable -> saved image path (or None if shown)
saved = woebin_plot(bins, save_dir='plots/woe', show=False)
print(saved.get('age'))  # e.g., 'plots/woe/woe_age.png'
```
