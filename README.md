# scorecardpl

A lightweight, pure-Python credit scoring toolkit inspired by `scorecardpy`.

Highlights:
- Quantile and categorical binning with WOE/IV
- WOE transform and plotting
- Simple variable filtering utilities
- Logistic Regression model and scorecard points mapping
- KS/AUC evaluation and basic plots

This is a minimal, developer-friendly implementation intended as a starting point.

## Quickstart

```python
import polars as pl
from scorecardpl import split_df, var_filter, woebin, woebin_ply, perf_eva, scorecard, scorecard_ply

# sample data (Polars)
df = pl.DataFrame({
    'y': [1,0,0,1,0,1,0,0,1,0]*100,
    'age': [23,45,36,52,41,29,33,60,38,27]*100,
    'inc': [3.2, 7.1, 5.4, 9.6, 6.1, 4.0, 5.1, 10.2, 5.9, 3.8]*100,
    'job': ['A','B','A','C','B','A','B','C','A','B']*100,
})

train, valid = split_df(df, y='y', test_size=0.3, random_state=42)
xs = ['age','inc','job']
train = var_filter(train, y='y', x=xs)

# Numeric binning method can be 'quantile' (default), 'equal_width', 'tree', 'chi2', 'isotonic', or 'custom' via breaks
bins = woebin(train, y='y', x=xs, bins=5, method='tree', tree_params={'min_leaf_frac': 0.05})
# Chi-square supervised merging
bins = woebin(train, y='y', x=xs, bins=6, method='chi2', chi2_params={'init_bins': 40})
# Isotonic supervised monotonic binning
bins = woebin(train, y='y', x=xs, bins=6, method='isotonic')
train_woe = woebin_ply(train, bins)
valid_woe = woebin_ply(valid, bins)

# fit LR on WOE features
sc = scorecard(bins, y='y', data=train_woe)

# evaluate (save plots to files instead of showing)
X_valid = valid_woe.select([c for c in valid_woe.columns if c.endswith('_woe')]).to_numpy()
pred = sc.model.predict_proba(X_valid)[:, 1]
perf = perf_eva(valid_woe['y'], pred, plot='both', save_prefix='examples/plots/quickstart')
print(perf)

# score mapping and application
card = sc.points_map
scores = scorecard_ply(valid_woe, card)
print(scores)
```

## Advanced Binning Examples

```python
# Monotonic WOE enforcement for numeric bins (after supervised tree splits)
bins = woebin(
    train,
    y='y', x=['age','inc','job'],
    bins=6,
    method='tree',
    tree_params={'min_leaf_frac': 0.03},
    monotonic='auto',         # or 'increasing' / 'decreasing'
    monotonic_min_bins=3,
)

# Supervised categorical merging to at most 3 bins using target-rate adjacency
bins = woebin(
    train,
    y='y', x=['age','inc','job'],
    bins=6,
    method='chi2',            # numeric: chi-square adjacent merge
    chi2_params={'init_bins': 40},
    cat_max_bins=3,
    cat_method='target_rate', # or 'chi2' / 'frequency'
)

# Isotonic (monotonic supervised) numeric binning
bins = woebin(
    train,
    y='y', x=['age','inc','job'],
    bins=6,
    method='isotonic',
)
```

See inline docstrings for more details.

Documentation
- Hosted docs (Read the Docs): configure with `.readthedocs.yaml` and the `docs/` folder. Build locally with:
  - `pip install -e .[docs] -r docs/requirements.txt`
  - `sphinx-build -b html docs/ docs/_build/html`
  - Optional live reload: `sphinx-autobuild docs/ docs/_build/html`

SHAP-based scorecards
- Build a scorecard from any sklearn classifier (e.g., RandomForest, XGBoost/LightGBM via sklearn wrappers) using SHAP values:
  - `pip install shap` (and optionally `xgboost lightgbm`)
  - See `examples/shap_example.py` and docs: `docs/shap-scorecard.md`


## Compare Binning Methods

Run a script that fits models using multiple binning strategies and saves ROC/KS plots:

```
python examples/compare_binning.py
```

Plots are saved under `examples/plots/` as `<method>_roc.png` and `<method>_ks.png`.
