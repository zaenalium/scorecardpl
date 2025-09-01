# SHAP-based Scorecard

You can build a scorecard from any scikit-learn classifier using SHAP values.

This keeps the binning/WOE transform pipeline but derives per-bin points from
mean SHAP contributions rather than logistic-regression coefficients.

## Install

Install SHAP (and optionally a GBM library):

```
pip install shap
# optional
pip install xgboost lightgbm
```

## Usage

```python
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from scorecardpl import woebin, woebin_ply, scorecard_shap, scorecard_ply

df = pl.read_parquet('data.parquet')
bins = woebin(df, y='y', x=['age','inc','job'], bins=6)

# transform; keep_bins=True (default) so we can join points by bin
df_w = woebin_ply(df, bins)

clf = RandomForestClassifier(n_estimators=200, random_state=7)
sc_shap = scorecard_shap(bins, y='y', data=df_w, estimator=clf, shap_sample_n=5000)

scores = scorecard_ply(df_w, sc_shap.points_map)
print(scores.head())
```

Notes:
- `shap_sample_n` limits rows used for SHAP value computation; increase for more precision.
- The output `points_map` can be applied the same way as the standard scorecard.
- If you use GBM (XGBoost/LightGBM), pass the sklearn wrapper (e.g., `XGBClassifier` or `LGBMClassifier`).

Example notebooks:
- `notebooks/german_credit_shap_xgboost.ipynb`
- `notebooks/german_credit_shap_randomforest.ipynb`

## Compatibility

- SHAP versions differ in the shape of classifier attributions:
  - Older: `(n_samples, n_classes, n_features)`; Newer (>=0.48): `(n_samples, n_features, n_outputs)`.
  - `scorecardpl` normalizes these shapes internally to `(n_samples, n_features)` and uses the positive class.
- For tree models, `TreeExplainer(model_output='log_odds')` is attempted first; if unsupported, a generic `Explainer` with a logit link is used so contributions are on the log-odds scale.
- Base values are handled across SHAP versions and converted to a single intercept on the log-odds scale for the scorecard bias points.
