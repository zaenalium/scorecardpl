# Getting Started

## Installation

scorecardpl targets Python 3.8+.

Install from source (editable):

```
pip install -e .
```

To build and serve the documentation (MkDocs):

```
pip install -r docs/requirements-mkdocs.txt
mkdocs serve -a 127.0.0.1:8000
```

Key runtime dependencies: polars, numpy, pandas, scikit-learn, matplotlib.

## Concepts

- WOE (Weight of Evidence) and IV (Information Value) for binary classification
- Binning strategies for numeric/categorical variables
- Transforming raw data to WOE features for logistic regression
- Mapping model coefficients to a scorecard (points) with PDO and base score
