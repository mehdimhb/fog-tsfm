"""Per-head cross-attention maps for one grid run, averaged over a split.

Works for any combo and FM subset, so the figure can show whichever model is
being reported.

    python tools/grid/plot_cross_attention.py --combo p0123_s2 --fm UM --split test
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "data"))
from src import data, model, task  # noqa: E402
from sensor_labels import PLACEMENT_LABELS, SENSOR_GROUP_CHANNELS  # noqa: E402

FM_NAMES = {"U": "unishape", "M": "mantis", "N": "nutime"}
DISPLAY = {"unishape": "UniShape", "mantis": "MantisV2", "nutime": "NuTime"}
AXIS = ["acc-x", "acc-y", "acc-z", "gyr-x", "gyr-y", "gyr-z"]


def channel_names(combo: str) -> list[str]:
    slots, sensor = combo.split("_")
    return [f"{PLACEMENT_LABELS[int(p)]}-{AXIS[c]}" for p in slots[1:] for c in SENSOR_GROUP_CHANNELS[sensor]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default="p0123_s2")
    ap.add_argument("--fm", default="UM")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out", default="cross_attention_um.pdf")
    args = ap.parse_args()

    mods = [FM_NAMES[k] for k in args.fm]
    name = f"combo_{args.combo}_{args.fm}"
    run = ROOT / "experiments/Combination_Ablation/Comb_Ensembles" / name
    best = json.loads((run / f"{name}_results.json").read_text())["best_epoch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = data.FeatureData(lmdb_path="datasets/new_lmdb_features", combo=args.combo,
                               models=mods, num_classes=2, one_hot_labels=True)
    dm = task.DataModule(data=dataset, batch_size=1024, num_workers=0, shuffle=False)
    dm.setup(stage="test")
    loader = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()

    net = model.Ensemble.load(str(ROOT / f"configs/ablation/{name}/model.json"),
                              str(run / "checkpoints" / f"epoch={best}.ckpt"))
    net.to(device).eval()

    sums, counts = {}, {}
    for mod in mods:
        mha = net.attns[mod].cross_attn
        orig = mha.forward
        mha.forward = (lambda f: lambda *a, **k: f(*a, **{**k, "average_attn_weights": False}))(orig)

        def hook(_m, _i, out, mod=mod):
            w = out[1].detach().cpu()
            sums[mod] = sums.get(mod, 0) + w.sum(0).numpy()
            counts[mod] = counts.get(mod, 0) + w.shape[0]
        mha.register_forward_hook(hook)

    with torch.no_grad():
        for batch in loader:
            net({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
    attn = {m: sums[m] / counts[m] for m in mods}

    heads, nch = attn[mods[0]].shape[:2]
    total = len(mods) * nch
    mega = np.full((heads, total, total), np.nan)  # NaN keeps intra-model blocks empty
    for r, mod in enumerate(mods):
        for j, other in enumerate(m for m in mods if m != mod):
            c = mods.index(other)
            mega[:, r * nch:(r + 1) * nch, c * nch:(c + 1) * nch] = attn[mod][:, :, j * nch:(j + 1) * nch]

    labels = channel_names(args.combo) * len(mods)
    assert len(labels) == total, (len(labels), total)
    vmin, vmax = np.nanmin(mega), np.nanmax(mega)
    fig, axes = plt.subplots(2, 2, figsize=(22, 20), sharex=True)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    centers = [j * nch + nch / 2 for j in range(len(mods))]
    for i, ax in enumerate(axes.flat[:heads]):
        row, col = divmod(i, 2)
        sns.heatmap(mega[i], cmap="magma", ax=ax, cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                    xticklabels=labels if row == 1 else False, yticklabels=labels if col == 0 else False,
                    vmin=vmin, vmax=vmax, rasterized=True)
        for j in range(1, len(mods)):
            ax.axhline(j * nch, color="white", linewidth=2); ax.axvline(j * nch, color="white", linewidth=2)
        ax.set_title(f"Head {i + 1}", fontsize=18)
        if col == 0:
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
            for c, m in zip(centers, mods):
                ax.text(-0.16, c, DISPLAY[m], transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=20, fontweight="bold", rotation=90)
        if row == 1:
            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            for c, m in zip(centers, mods):
                ax.text(c, -0.16, DISPLAY[m], transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=20, fontweight="bold")
    plt.subplots_adjust(wspace=0.05, hspace=0.08, right=0.9)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out} (epoch={best}, split={args.split}, heads={heads}, channels={nch})")


if __name__ == "__main__":
    main()
