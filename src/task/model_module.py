import torch
from lightning.pytorch import LightningModule

from ..model import BaseModel
from ..loss import CompositeLoss
from ..metric import MetricSet
from ..optimizer import BaseOptimizer


class ModelModule(LightningModule):
    def __init__(
        self,
        model: BaseModel,
        loss: CompositeLoss,
        metric: MetricSet,
        optimizer: BaseOptimizer,
    ):
        super().__init__()

        self.model = model
        self.optimizers_config = optimizer

        self.train_loss = loss(stage="train")
        self.train_metric = metric(stage="train")

        self.val_loss = loss(stage="val")
        self.val_metric = metric(stage="val")

        self.test_loss = loss(stage="test")
        self.test_metric = metric(stage="test")

        # Per-window scores from the most recent val/test pass, in dataloader
        # order. Summary metrics are computed at a fixed 0.5 cut, so without
        # these the operating point can never be revisited: a threshold sweep
        # cannot be recovered from a point metric once the run has exited.
        self._val_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._test_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.val_predictions: dict[str, torch.Tensor] | None = None
        self.test_predictions: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _score_and_label(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive-class probability and integer label, detached onto the CPU.

        Class index 1 is the positive class, matching how Sensitivity,
        Specificity and PosF1Score read their per-class results.
        """
        logits = output["logits"].detach().float()
        labels = output["labels"].detach()
        if labels.ndim == 2:
            labels = labels.argmax(dim=-1)
        if logits.ndim == 2 and logits.shape[-1] >= 2:
            scores = torch.softmax(logits, dim=-1)[:, 1]
        else:
            scores = torch.sigmoid(logits.reshape(len(logits)))
        return scores.cpu(), labels.reshape(-1).long().cpu()

    @staticmethod
    def _drain(buffer: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor] | None:
        if not buffer:
            return None
        scores = torch.cat([score for score, _ in buffer])
        labels = torch.cat([label for _, label in buffer])
        buffer.clear()
        return {"score": scores, "label": labels}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self.forward(batch)
        loss = self.train_loss.step(output)
        self.train_metric.step(output)
        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute_and_reset()
        metric = self.train_metric.compute_and_reset()
        self.log_dict(loss | metric)

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self.forward(batch)
        loss = self.val_loss.step(output)
        self.val_metric.step(output)
        self._val_buffer.append(self._score_and_label(output))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_buffer.clear()

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute_and_reset()
        metric = self.val_metric.compute_and_reset()
        self.log_dict(loss | metric)
        self.val_predictions = self._drain(self._val_buffer)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self.forward(batch)
        loss = self.test_loss.step(output)
        self.test_metric.step(output)
        self._test_buffer.append(self._score_and_label(output))
        return loss

    def on_test_epoch_start(self) -> None:
        self._test_buffer.clear()

    def on_test_epoch_end(self) -> None:
        loss = self.test_loss.compute_and_reset()
        metric = self.test_metric.compute_and_reset()
        self.log_dict(loss | metric)
        self.test_predictions = self._drain(self._test_buffer)

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        return self.optimizers_config.setup(self, total_steps)
