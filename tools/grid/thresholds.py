"""Equal-error-rate operating point, chosen on validation scores only."""
import numpy as np


def confusion(labels: np.ndarray, decision: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero(decision & (labels == 1)))
    fp = int(np.count_nonzero(decision & (labels == 0)))
    fn = int(np.count_nonzero(~decision & (labels == 1)))
    tn = int(np.count_nonzero(~decision & (labels == 0)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "pos_f1": 2 * precision * sensitivity / max(precision + sensitivity, 1e-12),
    }


def choose_threshold(val_labels: np.ndarray, val_scores: np.ndarray) -> float:
    """Threshold where validation sensitivity and specificity are closest."""
    grid = np.unique(np.quantile(val_scores, np.linspace(0.001, 0.999, 999)))
    best, best_value = 0.5, -np.inf
    for threshold in grid:
        metrics = confusion(val_labels, val_scores > threshold)
        value = -abs(metrics["sensitivity"] - metrics["specificity"])
        if value > best_value:
            best_value, best = value, float(threshold)
    return best
