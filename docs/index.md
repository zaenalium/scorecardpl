# scorecardpl Documentation

scorecardpl is a lightweight, Polars-first credit scoring toolkit inspired by scorecardpy.

- Efficient WOE/IV binning for numeric and categorical variables
- Multiple numeric binning methods (quantile, equal-width, tree, chi2, isotonic, custom)
- Monotonic WOE enforcement, supervised categorical merging
- WOE transform and scorecard mapping with streaming and lazy pipelines
- Evaluation (AUC/KS) with plot saving

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
quickstart
advanced-binning
large-scale-guide
shap-scorecard
api/index
```
