"""Collect the full FM-subset x sensor grid (7 x 45) into one table.

Every run is scored from its saved per-window predictions. Threshold metrics
use the EER threshold chosen on validation (thresholds.py), so the
test set only ever contributes the reported numbers, never a choice.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "data"))
from thresholds import choose_threshold, confusion  # noqa: E402
from sensor_labels import describe_combo  # noqa: E402

RUNS = ROOT / "experiments" / "Combination_Ablation" / "Comb_Ensembles"
FMS = ["U", "M", "N", "UM", "UN", "MN", "UMN"]


def run_dir(combo: str, fm: str) -> Path:
    return RUNS / f"combo_{combo}_{fm}"


def score(path: Path) -> dict:
    v = pd.read_csv(path / "predictions" / "val_predictions.csv")
    t = pd.read_csv(path / "predictions" / "test_predictions.csv")
    th = choose_threshold(v.label.values, v.score.values)
    tc = confusion(t.label.values, t.score.values > th)
    return {
        "val_auroc": roc_auc_score(v.label, v.score),
        "val_auprc": average_precision_score(v.label, v.score),
        "test_auroc": roc_auc_score(t.label, t.score),
        "test_auprc": average_precision_score(t.label, t.score),
        "test_f1": tc["pos_f1"], "test_acc": tc["accuracy"],
        "test_sens": tc["sensitivity"], "test_spec": tc["specificity"],
    }


def main() -> None:
    combos = sorted(p.name[len("combo_"):-len("_UMN")] for p in RUNS.glob("combo_*_UMN"))
    rows = []
    for combo in combos:
        for fm in FMS:
            rows.append({"combo": combo, "fm": fm, "sensors": describe_combo(combo),
                         "n_placements": len(combo.split("_")[0]) - 1,
                         "modality": combo.split("_")[1], **score(run_dir(combo, fm))})
    df = pd.DataFrame(rows)
    assert len(df) == 7 * 45, len(df)
    out = ROOT / "results" / "grid_7x45.csv"
    df.round(4).to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} runs)")


if __name__ == "__main__":
    main()
