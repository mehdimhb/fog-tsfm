"""20-s moving-block bootstrap CIs for test AUROC/AUPRC of chosen grid runs.

Windows overlap by 75 %, so resampling single windows would understate the
variance; blocks of 40 consecutive windows (20 s) keep neighbours together.
Subjects are still not resampled, so these intervals remain optimistic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_grid import run_dir  # noqa: E402

SOTA_AUROC = 0.9090
RUNS = [("p0123_s2", "UM"), ("p0123_s2", "UMN"), ("p023_s0", "UM"), ("p013_s2", "UMN")]
B, L = 2000, 40


def main() -> None:
    preds = {r: pd.read_csv(run_dir(*r) / "predictions" / "test_predictions.csv") for r in RUNS}
    y = preds[RUNS[0]].label.values
    assert all((p.label.values == y).all() for p in preds.values())
    n, rng = len(y), np.random.default_rng(0)
    au = {r: [] for r in RUNS}; ap = {r: [] for r in RUNS}
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + L) for s in rng.integers(0, n - L + 1, int(np.ceil(n / L)))])[:n]
        if y[idx].min() == y[idx].max():
            continue
        for r in RUNS:
            s = preds[r].score.values[idx]
            au[r].append(roc_auc_score(y[idx], s)); ap[r].append(average_precision_score(y[idx], s))
    for r in RUNS:
        a, p = np.array(au[r]), np.array(ap[r])
        print(f"{r[1]:>3} {r[0]:<9} AUROC {roc_auc_score(y, preds[r].score):.4f} "
              f"[{np.percentile(a, 2.5):.4f}, {np.percentile(a, 97.5):.4f}]  "
              f"AUPRC {average_precision_score(y, preds[r].score):.4f} "
              f"[{np.percentile(p, 2.5):.4f}, {np.percentile(p, 97.5):.4f}]  "
              f"P(AUROC<=SOTA)={np.mean(a <= SOTA_AUROC):.3f}")
    d = np.array(au[RUNS[0]]) - np.array(au[RUNS[1]])
    print(f"UM - UMN (all sensors) test AUROC: {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
