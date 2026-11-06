import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from src import data, loss, metric, model, optimizer, task

torch.set_float32_matmul_precision('high')

# Map short keys to the model lists
ENSEMBLE_MAP = {
    'U': ['unishape'],
    'M': ['mantis'],
    'N': ['nutime'],
    'UM': ['unishape', 'mantis'],
    'UN': ['unishape', 'nutime'],
    'MN': ['mantis', 'nutime'],
    'UMN': ['unishape', 'mantis', 'nutime']
}

def get_best_epoch(csv_path: Path) -> int:
    """Reads the CSV logs and finds the epoch with the highest average validation score."""
    df = pd.read_csv(csv_path)

    # Filter only validation rows
    val_cols = [c for c in df.columns if c.startswith('val/metric/')]
    df_val = df.dropna(subset=val_cols).copy()

    # Calculate a composite score (average of all validation metrics)
    df_val['composite_score'] = df_val[val_cols].mean(axis=1)

    best_row = df_val.loc[df_val['composite_score'].idxmax()]
    return int(best_row['epoch'])

def build_task(combo: str, ensemble_key: str = 'UMN', n_classes: int = 2):
    """Assemble the model, data and Task for one placement/sensor combination and ensemble."""
    # Append the ensemble key to version name so results save in distinct directories
    version_name = f"combo_{combo}_{ensemble_key}"
    active_models = ENSEMBLE_MAP[ensemble_key]

    p_part, s_part = combo.split('_')
    num_placements = len(p_part) - 1
    num_sensor_axes = 6 if s_part == 's2' else 3
    calculated_num_channels = num_placements * num_sensor_axes

    # Pass the parameterized active_models
    dataset = data.FeatureData(
        lmdb_path='datasets/new_lmdb_features',
        combo=combo,
        models=active_models,
        num_classes=n_classes,
        one_hot_labels=True
    )

    classifier = model.Ensemble(
        active_modalities=active_models,
        num_channels=calculated_num_channels,
        dim=128,
        num_classes=n_classes,
        num_heads=4,
        dropout=0.3,
        token_mask_prob=0.2
    )

    class_loss = loss.ClassificationLoss()
    total = loss.CompositeLoss([class_loss], weights=[1.0])

    acc = metric.Accuracy(num_classes=n_classes)
    f1 = metric.PosF1Score(num_classes=n_classes)
    sens = metric.Sensitivity(num_classes=n_classes)
    spec = metric.Specificity(num_classes=n_classes)
    aup = metric.AUPRC(num_classes=n_classes)
    auroc = metric.AUROC(num_classes=n_classes)

    metric_set = metric.MetricSet(
        train_metrics=[acc], eval_metrics=[acc, f1, sens, spec, aup, auroc]
    )

    adam_optimizer = optimizer.AdamWLinearProbe(
        learning_rate=2.5e-4, weight_decay=0.04, min_lr=1e-6, warmup_steps=1000
    )

    task_config = task.TaskConfig(
        major_task="Combination_Ablation",
        minor_task="Comb_Ensembles",
        version=version_name,
        task_description=f"Ablation combo: {combo}, ensemble: {ensemble_key}",
        device="cuda:0",
        experiments_dir="experiments",
        batch_size=128,
        num_workers=2,
        shuffle=True,
    )

    the_task = task.Task(
        model=classifier, data=dataset, loss=total, metric=metric_set,
        optimizer=adam_optimizer, task_config=task_config,
    )

    return the_task, version_name, calculated_num_channels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', type=str, required=True, help="The combo key, e.g., 'p03_s0'")
    parser.add_argument('--ensemble', type=str, default='UMN', choices=list(ENSEMBLE_MAP.keys()),
                        help="The modality ensemble to use, e.g., 'UM'")
    args = parser.parse_args()

    combo = args.combo
    ensemble_key = args.ensemble
    the_task, version_name, calculated_num_channels = build_task(combo, ensemble_key)

    the_task.save_config(version_name, "configs/ablation")

    # 1. Train the model
    the_task.train(epochs=50)

    # 2. Locate metrics.csv reliably
    experiment_path = the_task._get_experiment_path()
    csv_path = experiment_path / version_name / "metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Failed to create metrics file at {csv_path}. Training likely crashed.")

    # 3. Get the best epoch
    best_epoch = get_best_epoch(csv_path)
    print(f"[{version_name}] Best epoch across all metrics: {best_epoch}")

    # Load the best checkpoint and evaluate
    checkpoint_dir = the_task._get_checkpoint_path()
    best_ckpt_path = checkpoint_dir / f"epoch={best_epoch}.ckpt"

    accelerator, devices = the_task._get_accelerator_and_devices()
    from lightning.pytorch import Trainer
    eval_trainer = Trainer(accelerator=accelerator, devices=devices, logger=False)

    val_results = eval_trainer.validate(model=the_task.model_module, datamodule=the_task.data_module, ckpt_path=best_ckpt_path)
    val_predictions = the_task.model_module.val_predictions

    test_results = eval_trainer.test(model=the_task.model_module, datamodule=the_task.data_module, ckpt_path=best_ckpt_path)
    test_predictions = the_task.model_module.test_predictions

    # 4. Save results to JSON (added ensemble key)
    results = {
        "combo": combo,
        "ensemble": ensemble_key,
        "num_channels": calculated_num_channels,
        "best_epoch": best_epoch,
        "validation": val_results[0] if val_results else {},
        "test": test_results[0] if test_results else {}
    }

    results_file = experiment_path / version_name / f"{version_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # 5. Save the per-window scores
    predictions_dir = experiment_path / version_name / "predictions"
    for split, predictions, dataset in (
        ("val", val_predictions, the_task.data_module.data.val_ds),
        ("test", test_predictions, the_task.data_module.data.test_ds),
    ):
        written = task.write_predictions(
            predictions_dir / f"{split}_predictions.csv",
            predictions,
            sample_keys=getattr(dataset, "keys", None),
        )
        if written is None:
            print(f"[{version_name}] WARNING: no {split} predictions were captured.")
        else:
            print(f"[{version_name}] wrote {written}")