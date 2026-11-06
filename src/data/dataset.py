"""Leakage-safe train/validation/test loader for processed_windows.npz.

When ``base.py`` cannot be imported, the class falls back to the standard PyTorch
Dataset base so the split logic can still be tested independently.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .base import BaseDataset, define_dataset
except ImportError:  # Standalone verification outside the package.
    from torch.utils.data import Dataset as BaseDataset

    def define_dataset(dataset_class):
        return dataset_class


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return unique values in first-appearance order without sorting IDs."""
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def _read_csv_manifest(
    all_groups: np.ndarray,
    manifest_path: str | Path,
) -> dict[str, np.ndarray]:
    """Read the exact train/val/test split from a CSV manifest.

    Expects the CSV to contain 'subjectID' and 'split' columns.
    """
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Split manifest {manifest_path} is empty.")

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in rows:
        try:
            # Strip whitespace in case of minor formatting anomalies in the CSV
            group = str(row["subjectID"]).strip()
            split_val = str(row["split"]).lower().strip()
        except KeyError as e:
            raise KeyError(
                f"Missing expected column {e} in {manifest_path}. "
                f"Found headers: {list(rows[0].keys())}"
            )

        if split_val in splits:
            splits[split_val].append(group)
        elif split_val: # Warn or handle unexpected splits if they aren't empty
            raise ValueError(f"Unknown split partition '{split_val}' for subject {group}. Expected train, val, or test.")

    dtype = np.asarray(all_groups).dtype
    return {
        "train": np.asarray(sorted(splits["train"]), dtype=dtype),
        "val": np.asarray(sorted(splits["val"]), dtype=dtype),
        "test": np.asarray(sorted(splits["test"]), dtype=dtype),
    }


def build_manifest_split(
    group_ids: np.ndarray,
    labels: np.ndarray,
    split_csv_path: str | Path = "split_11_5_6.csv",
) -> tuple[dict[str, np.ndarray], str]:
    """Return the train/val/test partition directly from the CSV manifest."""
    group_ids = np.asarray(group_ids)
    labels = np.asarray(labels)
    if group_ids.ndim != 1 or labels.ndim != 1 or len(group_ids) != len(labels):
        raise ValueError("group_ids and labels must be aligned one-dimensional arrays.")

    unique_groups = _ordered_unique(group_ids)
    result = _read_csv_manifest(unique_groups, split_csv_path)
    source = f"frozen_csv_manifest_{Path(split_csv_path).name}"

    sets = {name: set(values.astype(str).tolist()) for name, values in result.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("Subject overlap detected across train/val/test sets in the CSV split manifest.")

    return result, source


class _Dataset(BaseDataset):
    def __init__(
        self,
        data_path,
        mode,
        split_csv="split_11_5_6.csv",
        transform=None,
        target_transform=None,
        return_masks=False,
        as_tensor=True,
        require_labels=True,
    ):
        """Load a deterministic subject-grouped train/val/test partition from CSV.

        Parameters
        ----------
        data_path:
            Path to ``processed_windows.npz``.
        mode:
            Exactly one of ``train``, ``val``, or ``test``.
        split_csv:
            Path to the CSV file defining the split (defaults to split_11_5_6.csv).
        require_labels:
            Defaults to True. Supervised train/val/test loading fails visibly if
            any window lacks a label. Set False only for a separately declared
            inference workflow, and also request masks so label availability is
            visible to the caller.
        """

        self.npz_path = str(Path(data_path))
        self.mode = str(mode)
        self.split_csv = split_csv
        self.transform = transform
        self.target_transform = target_transform
        self.return_masks = bool(return_masks)
        self.as_tensor = bool(as_tensor)
        self.require_labels = bool(require_labels)

        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"Mode must be 'train', 'val', or 'test'. Got {self.mode!r}")
        if not self.require_labels and not self.return_masks:
            raise ValueError(
                "require_labels=False requires return_masks=True so label availability is visible."
            )

        required = {"signals", "y", "group_id", "label_available"}
        if self.return_masks:
            required.add("valid_mask")
        with np.load(self.npz_path, allow_pickle=False) as data:
            missing = sorted(required - set(data.files))
            if missing:
                raise KeyError(f"processed_windows.npz is missing required keys: {missing}")
            all_x = np.asarray(data["signals"], dtype=np.float32)
            all_label = np.asarray(data["y"])
            # The npz still calls the array "group_id", but it maps to "subjectID" in the CSV
            all_group_ids = np.asarray(data["group_id"])
            all_label_mask = np.asarray(data["label_available"], dtype=bool)
            all_valid_mask = (
                np.asarray(data["valid_mask"], dtype=bool)
                if self.return_masks
                else None
            )

        count = len(all_x)
        if all_x.ndim != 4 or tuple(all_x.shape[1:]) != (4, 6, 120):
            raise ValueError(f"signals must have shape (N,4,6,120); got {all_x.shape}")
        if all_label.shape != (count,) or all_group_ids.shape != (count,):
            raise ValueError("y and group_id must each have shape (N,).")
        if all_label_mask.shape != (count,):
            raise ValueError("label_available must have shape (N,).")
        if all_valid_mask is not None and all_valid_mask.shape != all_x.shape:
            raise ValueError("valid_mask must exactly match signals.")
        if self.require_labels and not bool(all_label_mask.all()):
            missing_count = int((~all_label_mask).sum())
            raise ValueError(
                f"Supervised loading requires every window label; {missing_count} are unavailable."
            )

        split_groups, split_source = build_manifest_split(
            all_group_ids,
            all_label,
            split_csv_path=self.split_csv,
        )
        selected_groups = split_groups[self.mode]
        indices = np.flatnonzero(np.isin(all_group_ids, selected_groups))
        if not len(indices):
            raise AssertionError(f"Loaded {self.mode} partition contains no windows.")

        self.indices = indices
        self.group_id = all_group_ids[indices]
        self.x = all_x[indices]
        self.label = all_label[indices]
        self.label_mask = all_label_mask[indices]
        if all_valid_mask is not None:
            self.valid_mask = all_valid_mask[indices]
        self.split_groups = split_groups
        self.selected_groups = selected_groups
        self.split_audit = {
            "mode": self.mode,
            "split_source": split_source,
            "manifest_path": str(self.split_csv),
            "window_count": int(len(indices)),
            "selected_group_count": int(len(selected_groups)),
            "train_subject_count": int(len(split_groups["train"])),
            "val_subject_count": int(len(split_groups["val"])),
            "test_subject_count": int(len(split_groups["test"])),
            "subject_overlap": 0,
        }

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        frame = self.x[index]
        label = self.label[index]
        valid_mask = self.valid_mask[index] if self.return_masks else None

        if self.transform is not None:
            frame = self.transform(frame)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.as_tensor:
            frame = torch.as_tensor(frame, dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.long)
            if valid_mask is not None:
                valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
            label_mask = torch.as_tensor(self.label_mask[index], dtype=torch.bool)
        else:
            frame = np.asarray(frame, dtype=np.float32)
            label = np.asarray(label, dtype=np.int64)
            if valid_mask is not None:
                valid_mask = np.asarray(valid_mask, dtype=bool)
            label_mask = np.asarray(self.label_mask[index], dtype=bool)

        if frame.ndim != 3 or tuple(frame.shape[:2]) != (4, 6):
            raise ValueError("A transform changed the expected (4,6,T) signal structure.")
        placement_count, channel_count, time_count = frame.shape
        frame = frame.reshape(placement_count, channel_count, time_count)
        outputs = [frame, label]

        if self.return_masks:
            if valid_mask is None or valid_mask.shape != (4, 6, time_count):
                raise ValueError("valid_mask no longer matches the transformed signal shape.")
            valid_mask = valid_mask.reshape(placement_count, channel_count, time_count)
            outputs.extend([valid_mask, label_mask])
        return tuple(outputs)


Dataset = define_dataset(_Dataset)

__all__ = ["Dataset", "_Dataset", "build_manifest_split"]