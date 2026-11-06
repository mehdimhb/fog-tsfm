import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric

from .base import BaseLoss, define_loss


@define_loss
class ClassificationLoss(BaseLoss):
    def __init__(self, stage: str) -> None:
        super().__init__(loss_fn=MeanMetric(), name="Loss", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = output["logits"]
        labels = output["labels"]
        loss = F.cross_entropy(logits, labels)
        self.loss_fn.update(loss)
        return loss
