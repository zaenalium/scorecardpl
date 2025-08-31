from __future__ import annotations

from typing import Dict, Literal, Union, Optional

import numpy as np
import os
from sklearn import metrics
import matplotlib.pyplot as plt


def _to_numpy(x):
    try:
        # polars Series
        return x.to_numpy()
    except AttributeError:
        return np.asarray(x)


def ks_stat(y_true, y_score) -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def auc_score(y_true, y_score) -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    return float(metrics.roc_auc_score(y_true, y_score))


def perf_eva(
    y_true,
    y_score,
    plot: Union[bool, Literal['none','roc','ks','both']] = True,
    save_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate AUC and KS with optional plots.

    - plot: True/'both' to plot ROC and KS; 'roc' or 'ks' for a single plot; False/'none' for no plots.
    """
    # normalize plot flag
    plot_flag = 'both' if plot is True else ('none' if plot is False else plot)

    y_true_np = _to_numpy(y_true).astype(int)
    y_score_np = _to_numpy(y_score).astype(float)
    auc = auc_score(y_true_np, y_score_np)
    ks = ks_stat(y_true_np, y_score_np)
    out = {"auc": auc, "ks": ks}
    if plot_flag in ('roc','both'):
        fpr, tpr, _ = metrics.roc_curve(y_true_np, y_score_np)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--', alpha=0.5)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            os.makedirs(os.path.dirname(save_prefix) or '.', exist_ok=True)
            plt.savefig(f"{save_prefix}_roc.png", dpi=144)
            plt.close()
        else:
            plt.show()
    if plot_flag in ('ks','both'):
        order = np.argsort(y_score_np)
        y_ord = y_true_np[order]
        good = 1 - y_ord
        cum_bad = np.cumsum(y_ord) / max(y_ord.sum(), 1)
        cum_good = np.cumsum(good) / max(good.sum(), 1)
        plt.figure(figsize=(5,4))
        plt.plot(cum_bad, label='Cumulative Bad')
        plt.plot(cum_good, label='Cumulative Good')
        plt.title(f'KS = {ks:.3f}')
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            os.makedirs(os.path.dirname(save_prefix) or '.', exist_ok=True)
            plt.savefig(f"{save_prefix}_ks.png", dpi=144)
            plt.close()
        else:
            plt.show()
    return out
