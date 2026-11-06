from pathlib import Path
import json

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from ..base import Base
from ..data import BaseData
from ..loss import CompositeLoss
from ..metric import MetricSet
from ..model import BaseModel
from ..optimizer import BaseOptimizer
from .task_config import TaskConfig
from .model_module import ModelModule
from .data_module import DataModule


import csv
from lightning.pytorch.callbacks import Callback

class MetricsHistory(Callback):
    """Foolproof metric tracker that bypasses CSVLogger entirely."""
    def __init__(self, save_path):
        self.save_path = save_path
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        # Grab all logged metrics directly from Lightning's memory
        metrics = {"epoch": trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            metrics[k] = v.item() if hasattr(v, "item") else v

        self.history.append(metrics)

    def on_fit_end(self, trainer, pl_module):
        if not self.history:
            return

        # Collect all unique metric names
        keys = set()
        for m in self.history:
            keys.update(m.keys())

        # Sort keys so 'epoch' is the first column
        keys = ["epoch"] + [k for k in sorted(list(keys)) if k != "epoch"]

        # Write to CSV
        with open(self.save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)


class Task(Base):
    def __init__(
        self,
        model: BaseModel,
        data: BaseData,
        loss: CompositeLoss,
        metric: MetricSet,
        optimizer: BaseOptimizer,
        task_config: TaskConfig,
    ):
        super().__init__()

        self.model_module = ModelModule(
            model=model,
            loss=loss,
            metric=metric,
            optimizer=optimizer,
        )
        self.data_module = DataModule(
            data=data,
            batch_size=task_config["batch_size"],
            num_workers=task_config["num_workers"],
            shuffle=task_config["shuffle"]
        )
        self.model = model
        self.data = data
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.task_config = task_config

        self.capture_config()

    def _get_experiment_path(self) -> Path:
        experiment_path = Path(self.task_config["experiments_dir"])
        experiment_path = (
            experiment_path
            / self.task_config["major_task"]
            / self.task_config["minor_task"]
        )
        experiment_path.mkdir(parents=True, exist_ok=True)
        return experiment_path

    def _get_checkpoint_path(self) -> Path:
        experiment_path = self._get_experiment_path()
        checkpoint_path = experiment_path / self.task_config["version"] / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path

    def _get_accelerator_and_devices(self) -> tuple[str, int | list[int]]:
        device = self.task_config["device"]
        if device == "cpu":
            return "cpu", 1
        if device == "cuda":
            return "gpu", [0, 1]
        if device.startswith("cuda:"):
            try:
                return "gpu", [int(device.split(":", 1)[1])]
            except ValueError:
                pass
        raise ValueError(f"Unsupported device: {device}")

    def train(self, epochs: int) -> None:
        experiment_path = self._get_experiment_path()
        version_path = experiment_path / self.task_config["version"]

        # Path where our custom callback will save the metrics
        metrics_csv_path = version_path / "metrics.csv"

        tb_logger = TensorBoardLogger(
            save_dir=experiment_path,
            version=self.task_config["version"],
            sub_dir="logs",
            name="",
            default_hp_metric=False,
        )

        checkpoint_path = self._get_checkpoint_path()
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="epoch={epoch}",
            save_top_k=-1, # -1 saves EVERY epoch
            save_last=True,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=True
        )

        # Initialize our foolproof callback
        metrics_callback = MetricsHistory(save_path=metrics_csv_path)

        accelerator, devices = self._get_accelerator_and_devices()

        trainer = Trainer(
            max_epochs=epochs,
            logger=tb_logger, # We only need TensorBoard here now
            callbacks=[checkpoint, metrics_callback], # Add the custom callback
            accelerator=accelerator,
            devices=devices,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,
            log_every_n_steps=10
        )

        trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
        )

    @staticmethod
    def _save_object(
        obj, config_name: str, path: Path, extra_folder: str | None = None
    ) -> None:
        if extra_folder is not None:
            path = path / extra_folder

        if hasattr(obj, "config"):
            obj.save_config(config_name=config_name, save_dir=path)

    def _auto_save(self, obj, name: str, path: Path) -> None:
        """Recursively scan and save an object and its Base-inheriting children."""
        if not hasattr(obj, "config"):
            return

        # 1. Traverse and save nested objects first
        parameters = obj.config.get("parameters", {})
        for key, value in parameters.items():
            if hasattr(value, "config"):
                self._auto_save(value, name=key, path=path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if hasattr(item, "config"):
                        # Save in a subfolder named after the list key (e.g., 'losses')
                        sub_key = key
                        if key == 'losses':
                            sub_key = 'loss'
                        elif key == 'metrics':
                            sub_key = 'metric'
                        self._auto_save(item, name=f"{sub_key}_{i + 1}", path=path / key)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(v, "config"):
                        sub_key = key
                        if key == 'losses':
                            sub_key = 'loss'
                        elif key == 'metrics':
                            sub_key = 'metric'
                        self._auto_save(v, name=f"{sub_key}_{k}", path=path / key)

        # 2. Save the parent object itself now that children have valid paths
        if hasattr(obj, "save_config"):
            obj.save_config(config_name=name, save_dir=path)

    def _save(self, path: Path) -> None:
        # Automatically traverse all injected parameters captured in Task's config
        for key, value in self.config.get("parameters", {}).items():
            self._auto_save(value, name=key, path=path)

        self.base_save_config(config_name="task", save_dir=path)

    def save_config(
        self, config_name: str | None = None, save_dir: str | Path | None = None
    ) -> None:
        if config_name is not None and save_dir is not None:
            path = Path(save_dir) / config_name
        else:
            path = self._get_experiment_path() / self.task_config["version"] / "configs"
        self._save(path)

    @classmethod
    def from_config(
            cls, path: str | Path, new_parameters: dict | None = None
    ) -> 'Task':
        config = json.loads(Path(path).read_text())

        parameters = cls._load_params(config["parameters"])

        if new_parameters is not None:
            for name, parameter in new_parameters.items():
                parameters[name] = parameter

        return cls(**parameters)
