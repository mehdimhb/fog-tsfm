#!/usr/bin/env python
"""Validate one FoG-STAR-style sensor CSV and create model-ready windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "processor_config.json"


@dataclass(frozen=True)
class Recording:
    group_id: str
    recording_id: str
    source_rows: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    labels: np.ndarray
    label_available: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default="datasets/sensor_data.csv")
    parser.add_argument("--output", type=Path, default="datasets/processed_output2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pseudonym(prefix: str, value: str, salt: str) -> str:
    digest = hashlib.sha256((salt + "\0" + value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    signals = config.get("signal_columns")
    if not isinstance(signals, list) or len(signals) != 24 or len(set(signals)) != 24:
        raise ValueError("processor_config.json must declare 24 unique signal columns.")
    rate = float(config["sample_rate_hz"])
    if not math.isclose(rate, 60.0):
        raise ValueError("This NuTime/FoG-STAR package is locked to 60 Hz.")
    window_samples = int(round(rate * float(config["windows"]["duration_seconds"])))
    hop_samples = int(round(rate * float(config["windows"]["hop_seconds"])))
    if window_samples != 120 or hop_samples != 30:
        raise ValueError("This package requires 120-sample windows and 30-sample hops.")
    filter_cfg = config["filter"]
    if filter_cfg.get("short_valid_run_policy") != "error":
        raise ValueError("Short valid runs must fail visibly; silent fallback is prohibited.")
    return config


def numeric_column(frame: pd.DataFrame, name: str, *, allow_missing: bool) -> np.ndarray:
    source = frame[name]
    parsed = pd.to_numeric(source, errors="coerce")
    nonempty = source.notna() & source.astype(str).str.strip().ne("")
    malformed = nonempty & parsed.isna()
    if bool(malformed.any()):
        rows = np.flatnonzero(malformed.to_numpy())[:10].tolist()
        raise ValueError(f"Column {name!r} contains non-numeric text at zero-based rows {rows}.")
    values = parsed.to_numpy(np.float64)
    if np.isinf(values).any():
        rows = np.flatnonzero(np.isinf(values))[:10].tolist()
        raise ValueError(f"Column {name!r} contains infinity at zero-based rows {rows}.")
    if not allow_missing and np.isnan(values).any():
        rows = np.flatnonzero(np.isnan(values))[:10].tolist()
        raise ValueError(f"Column {name!r} is missing at zero-based rows {rows}.")
    return values


def validate_timestamp(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    timestamp_column: str,
    sample_rate: float,
    tolerance_fraction: float,
) -> dict[str, Any]:
    if timestamp_column not in frame.columns:
        return {"present": False, "used_as_model_input": False}
    values = numeric_column(frame.iloc[row_indices], timestamp_column, allow_missing=False)
    if len(values) > 1:
        differences = np.diff(values)
        expected = 1.0 / sample_rate
        if np.any(differences <= 0):
            raise ValueError("timestamp must increase strictly inside every recording.")
        bad = np.abs(differences - expected) > expected * tolerance_fraction
        if bool(bad.any()):
            raise ValueError(
                "timestamp increments do not match 60 Hz within the configured tolerance. "
                "This processor does not silently resample data."
            )
        observed = float(np.median(differences))
    else:
        observed = None
    return {
        "present": True,
        "used_as_model_input": False,
        "median_increment_seconds": observed,
    }


def filter_values(
    values: np.ndarray,
    valid: np.ndarray,
    config: Mapping[str, Any],
    recording_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    filter_cfg = config["filter"]
    if not bool(filter_cfg["enabled"]):
        return np.where(valid, values, 0.0).astype(np.float32), []
    sos = butter(
        int(filter_cfg["order"]),
        [float(filter_cfg["low_hz"]), float(filter_cfg["high_hz"])],
        btype="bandpass",
        fs=float(config["sample_rate_hz"]),
        output="sos",
    )
    result = np.zeros_like(values, dtype=np.float32)
    run_rows: list[dict[str, Any]] = []
    for channel in range(values.shape[1]):
        for run_index, (start, end) in enumerate(true_runs(valid[:, channel])):
            source = values[start:end, channel]
            try:
                filtered = sosfiltfilt(sos, source.astype(np.float64, copy=False))
            except ValueError as exc:
                raise ValueError(
                    f"Valid run is too short for gap-safe filtering: recording={recording_id}, "
                    f"channel_index={channel}, start={start}, end={end}, length={end-start}. "
                    "Collect a longer continuous run or explicitly redesign the preprocessing."
                ) from exc
            if not np.isfinite(filtered).all():
                raise FloatingPointError("Filtering produced a non-finite value.")
            result[start:end, channel] = filtered.astype(np.float32)
            run_rows.append(
                {
                    "recording_id": recording_id,
                    "channel_index": channel,
                    "run_index": run_index,
                    "start": start,
                    "end_exclusive": end,
                    "length_samples": end - start,
                }
            )
    if np.any(result[~valid] != 0) or not np.isfinite(result).all():
        raise AssertionError("Gap-safe filter/mask contract failed.")
    return result, run_rows


def prepare_recordings(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[list[Recording], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    signal_columns = list(config["signal_columns"])
    missing = [name for name in signal_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    identity_columns = list(config["identity_columns"])
    present_ids = [name for name in identity_columns if name in frame.columns]
    if present_ids and len(present_ids) != len(identity_columns):
        absent = [name for name in identity_columns if name not in frame.columns]
        raise ValueError(f"Use all identity columns together; missing: {absent}")
    if present_ids and frame[identity_columns].isna().any().any():
        raise ValueError("Identity columns cannot be missing.")

    label_name = str(config["optional_label_column"])
    label_present = label_name in frame.columns
    salt = str(config["privacy"]["pseudonym_salt"])

    # Updated Privacy Warning
    if not salt or salt.startswith("change-this"):
        privacy_warning = "Default pseudonym salt in use for recording IDs."
    else:
        privacy_warning = "Recording identifiers pseudonymized. Subject IDs are preserved in plaintext for CSV split compatibility."

    if present_ids:
        grouped = frame.groupby(identity_columns, sort=False, dropna=False).indices.items()
    else:
        grouped = [(('single_subject', 'single_session', 'single_task'), np.arange(len(frame)))]

    recordings: list[Recording] = []
    processed_frames: list[pd.DataFrame] = []
    timestamp_audits: list[dict[str, Any]] = []
    filter_runs: list[dict[str, Any]] = []
    for key, raw_indices in grouped:
        indices = np.asarray(raw_indices, dtype=np.int64)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Each recording must occupy one contiguous CSV block.")

        local = frame.iloc[indices]
        key_tuple = key if isinstance(key, tuple) else (key,)

        # Explicitly pull the subjectID without hashing it so it matches split manifests
        if "subjectID" in local.columns:
            subject = str(local["subjectID"].iloc[0]).strip()
        else:
            subject = str(key_tuple[0])

        full_key = "|".join(map(str, key_tuple))
        group_id = subject  # Retain plaintext subject ID
        recording_id = pseudonym("recording", full_key, salt)

        columns = [numeric_column(local, name, allow_missing=True) for name in signal_columns]
        values = np.column_stack(columns)
        valid = np.isfinite(values)
        timestamp = validate_timestamp(
            frame,
            indices,
            str(config["optional_timestamp_column"]),
            float(config["sample_rate_hz"]),
            float(config["timestamp_tolerance_fraction"]),
        )
        timestamp.update({"recording_id": recording_id, "row_count": len(indices)})
        timestamp_audits.append(timestamp)
        filtered, runs = filter_values(values, valid, config, recording_id)
        filter_runs.extend(runs)

        if label_present:
            raw_labels = numeric_column(local, label_name, allow_missing=True)
            label_available = np.isfinite(raw_labels)
            if np.any(label_available & ~np.isin(raw_labels, [0.0, 1.0])):
                raise ValueError("fog labels must be 0, 1, or blank.")
            labels = np.where(label_available, raw_labels, 0).astype(np.uint8)
        else:
            labels = np.zeros(len(indices), dtype=np.uint8)
            label_available = np.zeros(len(indices), dtype=bool)

        recordings.append(
            Recording(
                group_id=group_id,
                recording_id=recording_id,
                source_rows=indices,
                values=filtered,
                valid=valid,
                labels=labels,
                label_available=label_available,
            )
        )
        output = pd.DataFrame(filtered, columns=signal_columns)
        output[signal_columns] = output[signal_columns].where(valid, np.nan)
        output.insert(0, "relative_time_s", np.arange(len(indices)) / float(config["sample_rate_hz"]))
        output.insert(0, "sample_index", np.arange(len(indices), dtype=np.int64))
        output.insert(0, "recording_id", recording_id)
        output.insert(0, "group_id", group_id)
        if label_present:
            output[label_name] = pd.array(
                np.where(label_available, labels, np.nan), dtype="Float64"
            )
        processed_frames.append(output)

    processed = pd.concat(processed_frames, ignore_index=True)
    return recordings, processed, timestamp_audits, [
        {"privacy_message": privacy_warning}, *filter_runs
    ]


def build_windows(
    recordings: list[Recording], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rate = float(config["sample_rate_hz"])
    samples = int(round(rate * float(config["windows"]["duration_seconds"])))
    hop = int(round(rate * float(config["windows"]["hop_seconds"])))
    left = samples // 2
    rows: list[dict[str, Any]] = []
    x_items: list[np.ndarray] = []
    mask_items: list[np.ndarray] = []
    y_items: list[int] = []
    label_items: list[bool] = []
    group_items: list[str] = []
    recording_items: list[str] = []
    anchor_items: list[int] = []

    for recording in recordings:
        length = recording.values.shape[0]
        for anchor in range(0, length, hop):
            context = np.zeros((samples, 24), dtype=np.float32)
            context_mask = np.zeros((samples, 24), dtype=bool)
            source_start = max(0, anchor - left)
            source_end = min(length, anchor + (samples - left))
            target_start = source_start - (anchor - left)
            target_end = target_start + (source_end - source_start)
            context[target_start:target_end] = recording.values[source_start:source_end]
            context_mask[target_start:target_end] = recording.valid[source_start:source_end]

            counts = context_mask.sum(axis=0, keepdims=True)
            means = np.divide(
                np.where(context_mask, context, 0.0).sum(axis=0, keepdims=True, dtype=np.float64),
                counts,
                out=np.zeros((1, 24), dtype=np.float64),
                where=counts > 0,
            ).astype(np.float32)
            context = np.where(context_mask, context - means, 0.0).astype(np.float32)
            structured = context.T.reshape(4, 6, samples)
            structured_mask = context_mask.T.reshape(4, 6, samples)
            padded = source_start != anchor - left or source_end != anchor + (samples - left)

            x_items.append(structured)
            mask_items.append(structured_mask)
            y_items.append(int(recording.labels[anchor]))
            label_items.append(bool(recording.label_available[anchor]))
            group_items.append(recording.group_id)
            recording_items.append(recording.recording_id)
            anchor_items.append(anchor)
            rows.append(
                {
                    "window_id": len(rows),
                    "group_id": recording.group_id,
                    "recording_id": recording.recording_id,
                    "anchor_index": anchor,
                    "anchor_time_s": anchor / rate,
                    "fog": int(recording.labels[anchor]) if recording.label_available[anchor] else pd.NA,
                    "label_available": bool(recording.label_available[anchor]),
                    "valid_fraction": float(structured_mask.mean()),
                    "fully_valid_channel_count": int(structured_mask.all(axis=-1).sum()),
                    "is_boundary_padded": bool(padded),
                }
            )

    if not x_items:
        raise ValueError("The CSV produced no windows.")
    arrays = {
        "signals": np.stack(x_items).astype(np.float32),
        "valid_mask": np.stack(mask_items).astype(bool),
        "y": np.asarray(y_items, dtype=np.uint8),
        "label_available": np.asarray(label_items, dtype=bool),
        # Increased string length buffer slightly just in case plaintext IDs are long
        "group_id": np.asarray(group_items, dtype="U64"),
        "recording_id": np.asarray(recording_items, dtype="U36"),
        "anchor_index": np.asarray(anchor_items, dtype=np.int64),
        "sample_rate_hz": np.asarray(rate, dtype=np.float64),
        "channel_names": np.asarray(config["signal_columns"], dtype="U32"),
    }
    if arrays["signals"].shape[1:] != (4, 6, 120):
        raise AssertionError("Model-ready tensor boundary changed.")
    if arrays["signals"].dtype != np.float32 or arrays["valid_mask"].dtype != bool:
        raise AssertionError("Model-ready dtype contract changed.")
    if np.any(arrays["signals"][~arrays["valid_mask"]] != 0):
        raise AssertionError("Invalid/padded values must be exact zero below the mask.")
    return arrays, pd.DataFrame(rows)


def channel_statistics(
    frame: pd.DataFrame,
    processed: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in signal_columns:
        raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
        filtered = pd.to_numeric(processed[name], errors="coerce").to_numpy(np.float64)
        valid_raw = raw[np.isfinite(raw)]
        valid_filtered = filtered[np.isfinite(filtered)]
        rows.append(
            {
                "channel": name,
                "raw_valid_count": len(valid_raw),
                "missing_count": int(np.count_nonzero(~np.isfinite(raw))),
                "missing_fraction": float(np.mean(~np.isfinite(raw))),
                "raw_min": float(valid_raw.min()) if len(valid_raw) else np.nan,
                "raw_max": float(valid_raw.max()) if len(valid_raw) else np.nan,
                "raw_mean": float(valid_raw.mean()) if len(valid_raw) else np.nan,
                "raw_std": float(valid_raw.std()) if len(valid_raw) else np.nan,
                "filtered_min": float(valid_filtered.min()) if len(valid_filtered) else np.nan,
                "filtered_max": float(valid_filtered.max()) if len(valid_filtered) else np.nan,
                "filtered_mean": float(valid_filtered.mean()) if len(valid_filtered) else np.nan,
                "filtered_std": float(valid_filtered.std()) if len(valid_filtered) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def process(input_path: Path, output_path: Path, config_path: Path, force: bool) -> None:
    source = input_path.resolve()
    output = output_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    if source.suffix.lower() != ".csv":
        raise ValueError("Input must be a .csv file.")
    if output == source or output in source.parents:
        raise ValueError("Output must not replace or contain the source CSV.")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}. Use --force to replace it intentionally.")

    config = load_config(config_path.resolve())
    frame = pd.read_csv(source, low_memory=False)
    if frame.empty:
        raise ValueError("Input CSV contains no data rows.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"CSV contains duplicate column names: {duplicates}")

    recordings, processed, timestamp_audits, filter_audit = prepare_recordings(frame, config)
    arrays, metadata = build_windows(recordings, config)
    statistics = channel_statistics(frame, processed, list(config["signal_columns"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        processed.to_csv(temporary / "processed_data.csv", index=False, lineterminator="\n")
        metadata.to_csv(temporary / "window_metadata.csv", index=False, lineterminator="\n")
        statistics.to_csv(temporary / "channel_statistics.csv", index=False, lineterminator="\n")
        np.savez_compressed(temporary / "processed_windows.npz", **arrays)
        filter_runs = [row for row in filter_audit if "channel_index" in row]
        audit: dict[str, Any] = {
            "status": "technical_pass_data_preparation_only",
            "training_executed": False,
            "source_csv": source.name,
            "source_csv_sha256": sha256_file(source),
            "config_sha256": sha256_file(config_path.resolve()),
            "source_rows": int(len(frame)),
            "recording_count": len(recordings),
            "window_count": int(arrays["signals"].shape[0]),
            "model_ready_signal_shape": list(arrays["signals"].shape),
            "model_ready_mask_shape": list(arrays["valid_mask"].shape),
            "positive_labeled_windows": int(arrays["y"][arrays["label_available"]].sum()),
            "labeled_window_count": int(arrays["label_available"].sum()),
            "missing_signal_cells": int(sum((~recording.valid).sum() for recording in recordings)),
            "valid_filter_run_count": len(filter_runs),
            "minimum_filter_run_samples": min((row["length_samples"] for row in filter_runs), default=None),
            "sample_rate_hz": float(config["sample_rate_hz"]),
            "channel_order": list(config["signal_columns"]),
            "timestamp_audit": timestamp_audits,
            "privacy_message": filter_audit[0]["privacy_message"],
            "policies": {
                "interpolation": "none",
                "filtering": "gap-safe per contiguous valid recording/channel run",
                "window": "2 seconds centered on current sample every 0.5 seconds",
                "padding": "zero below false mask; every anchor retained",
                "window_mean_removal": True,
                "global_scaling": "not performed; must be fit on training participants only",
                "absolute_timestamp_model_input": False,
            },
        }
        for name in (
            "processed_data.csv",
            "processed_windows.npz",
            "window_metadata.csv",
            "channel_statistics.csv",
        ):
            audit.setdefault("output_sha256", {})[name] = sha256_file(temporary / name)
        write_json(temporary / "processing_audit.json", audit)

        with np.load(temporary / "processed_windows.npz", allow_pickle=False) as saved:
            if saved.files != list(arrays.keys()):
                raise AssertionError("Saved NPZ member order changed.")
            if saved["signals"].shape != arrays["signals"].shape:
                raise AssertionError("Saved NPZ shape verification failed.")
        if len(pd.read_csv(temporary / "window_metadata.csv")) != len(metadata):
            raise AssertionError("Saved window metadata coverage failed.")

        if output.exists():
            if not force:
                raise FileExistsError(output)
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "success",
        "output": str(output),
        "source_rows": len(frame),
        "recordings": len(recordings),
        "windows": int(arrays["signals"].shape[0]),
        "signal_shape": list(arrays["signals"].shape),
    }, indent=2))


def main() -> None:
    args = parse_args()
    process(args.input, args.output, args.config, args.force)


if __name__ == "__main__":
    main()