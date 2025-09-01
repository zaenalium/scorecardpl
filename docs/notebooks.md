# Example Notebooks

These example notebooks demonstrate end-to-end workflows with the German Credit dataset and `scorecardpl`.

- German Credit Analysis (Logistic Scorecard): `notebooks/german_credit_analysis.ipynb`
  - Load data, binning (chi2 + monotonic auto), logistic scorecard, AUC/KS, WOE plots.

- SHAP Scorecard (XGBoost): `notebooks/german_credit_shap_xgboost.ipynb`
  - Train XGBoost on WOE features and build a SHAP-derived scorecard; evaluate AUC/KS and compute PSI.

- SHAP Scorecard (RandomForest): `notebooks/german_credit_shap_randomforest.ipynb`
  - Train a RandomForest and build a SHAP-derived scorecard using the same pipeline.

- Calibration (Points → Probability): `notebooks/german_credit_calibration.ipynb`
  - Fit a calibrator that maps scorecard points to probabilities (logistic or isotonic) and evaluate.

- Drift / PSI Monitoring: `notebooks/german_credit_drift_psi.ipynb`
  - Compute Population Stability Index (PSI) between train and validation partitions per variable.

- Points Components: `notebooks/scorecard_components_demo.ipynb`
  - Show per-variable point contributions and total score using `scorecard_components`.

Notes:
- The notebooks download the UCI Statlog German Credit dataset at runtime.
- Plots are saved into a local `plots/` folder when applicable.
- SHAP-based notebooks require `shap` (`pip install shap`) and optionally GBMs (e.g., `xgboost`).

