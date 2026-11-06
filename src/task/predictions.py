"""Persist per-window scores so the operating point stays revisitable.

The metric stack reports Sensitivity, Specificity, Accuracy and Pos-F1 at a
fixed 0.5 cut, because MulticlassRecall and friends apply argmax internally.
On a 19.5 %-positive dataset that sits far toward the majority class: measured
across the sweep, mean test sensitivity 0.6253 against specificity 0.9156.

That is a reporting choice, not a model defect - AUROC and AUPRC are unaffected.
But a point metric cannot be re-expressed at a different threshold afterwards.
The curve it came from is gone as soon as the process exits, so the only way to
recover sensitivity at, say, the equal-error point is to have kept the scores.

This module writes them next to the metrics that were already being saved.
Selection then happens offline, in tools/grid/thresholds.py,
against the validation scores only - never against test.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def write_predictions(
    destination: str | Path,
    predictions: dict[str, torch.Tensor] | None,
    sample_keys: list[str] | None = None,
) -> Path | None:
    """Write ``score,label`` (plus ``sample_key`` when known) to a CSV.

    ``predictions`` is what ``ModelModule.val_predictions`` / ``.test_predictions``
    hold after a validate/test pass: scores in dataloader order.

    ``sample_keys`` comes from the dataset (``_FeatureDataset.keys``) and gives
    each row a stable identity, so scores can be joined back to the LMDB record -
    and through it to the subject id - without relying on row order.
    """
    if predictions is None:
        return None

    scores = predictions["score"].reshape(-1).tolist()
    labels = predictions["label"].reshape(-1).tolist()
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores against {len(labels)} labels")

    if sample_keys is not None and len(sample_keys) != len(scores):
        # Loud rather than silent: a mismatch means the rows and the keys are not
        # the same windows, and a quietly misaligned join is worse than no join.
        raise ValueError(
            f"{len(sample_keys)} dataset keys against {len(scores)} predictions. "
            "The dataloader order and the key list have diverged; refusing to "
            "write a mislabelled file."
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if sample_keys is None:
            writer.writerow(["score", "label"])
            writer.writerows(zip(scores, labels))
        else:
            writer.writerow(["sample_key", "score", "label"])
            writer.writerows(zip(sample_keys, scores, labels))
    return destination


__all__ = ["write_predictions"]
