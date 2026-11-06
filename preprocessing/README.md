# Simple FoG-STAR Sensor CSV Processor

This standalone package converts one continuous sensor CSV into validated,
gap-safe, model-ready FoG-STAR windows. It does **not** train a model.

## Required sensor columns

The CSV must contain these 24 columns in any column order:

```text
ankleL_acc_x, ankleL_acc_y, ankleL_acc_z
ankleL_gyro_x, ankleL_gyro_y, ankleL_gyro_z
ankleR_acc_x, ankleR_acc_y, ankleR_acc_z
ankleR_gyro_x, ankleR_gyro_y, ankleR_gyro_z
back_acc_x, back_acc_y, back_acc_z
back_gyro_x, back_gyro_y, back_gyro_z
wrist_acc_x, wrist_acc_y, wrist_acc_z
wrist_gyro_x, wrist_gyro_y, wrist_gyro_z
```

Accelerometer values are expected in `g`; gyroscope values are expected in
`degree/s`; the fixed sample rate is 60 Hz.

## Optional columns

- `subjectID`, `sessionID`, and `taskID`: use all three together when the CSV
  contains more than one physical recording.
- `fog`: current FoG label (`0` or `1`) for supervised data.
- `timestamp`: seconds. It is checked, then replaced by a recording-local time
  in the processed output. Absolute timestamp is never a model feature.

If none of the three identity columns is present, the whole file is treated as
one recording. If `fog` is absent, windows are still prepared for inference.

## Processing performed

```text
strict CSV/schema validation
-> retain missing samples in a Boolean mask
-> gap-safe fourth-order 0.5-20 Hz Butterworth filtering
-> 2-second windows (120 samples) every 0.5 seconds
-> boundary padding represented only by the mask
-> valid-sample mean removal inside each channel/window
-> save float32 (N,4,6,120) signals plus Boolean mask
```

No gap is interpolated. Filtering never crosses a missing interval or recording
boundary. Short valid runs that cannot be filtered cause a visible error rather
than a silent fallback. Global scaling is deliberately not applied: a training
pipeline must fit robust scaling only on its training participants.

## Outputs

- `processed_data.csv`: gap-safe filtered continuous signals and local time.
- `processed_windows.npz`: model-ready signals, validity mask, labels when
  available, and pseudonymized group/recording identifiers.
- `window_metadata.csv`: one audit row per 2-second context.
- `channel_statistics.csv`: raw/filtered ranges and missingness.
- `processing_audit.json`: source hash, shapes, counts, policy, and output
  hashes.

The NPZ tensor order is placement-major:

```text
(N, placement=4, channel=6, time=120)
placements: left ankle, right ankle, back, wrist
channels:   acc x/y/z, gyro x/y/z
```

## Usage

From the repository root:

```bash
uv run preprocessing/process_sensor_csv.py --input datasets/sensor_data.csv --output datasets/processed_output2
```

Add `--force` to replace an existing output folder.
