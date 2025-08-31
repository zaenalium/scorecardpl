# Calibration

scorecardpl provides simple calibration utilities to map scorecard points to probabilities.

## Methods

- Logistic (default): fits P(y=1) ~ sigmoid(a + b * points)
- Isotonic: non-parametric monotonic mapping from points to probability

## Usage

```python
from scorecardpl import (
  woebin, woebin_ply, scorecard, scorecard_ply,
  fit_points_calibrator, calibrate_scorecard_from_data,
  make_points_proba_fn, fit_scorecard_predictor
)

# assume bins and a points_map (from LR or SHAP)
df_w = woebin_ply(df, bins)
points = scorecard_ply(df_w, points_map)

# Logistic calibration
cal_log = fit_points_calibrator(df_w['y'], points, method='logistic')
proba = cal_log.points_to_proba(points)

# Isotonic calibration
cal_iso = fit_points_calibrator(df_w['y'], points, method='isotonic')
proba2 = cal_iso.points_to_proba(points)

# Convenience wrapper (transforms and fits):
cal = calibrate_scorecard_from_data(bins, points_map, y='y', data=df)

# Create a points->proba function
proba_fn = make_points_proba_fn(cal)

# Fit a deployable predictor from bins + points_map
cal2, predict_df = fit_scorecard_predictor(bins, points_map, y='y', data=df)
proba_df = predict_df(df)
```

For SHAP-based scorecards, use `scorecard_shap` to produce `points_map`, then calibrate
the resulting points with either method (isotonic often works well for complex models).
