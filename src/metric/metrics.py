import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassSpecificity,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassAveragePrecision
)
from .base import BaseMetric, define_metric


@define_metric
class Accuracy(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 2, ignore_index: int = -100) -> None:
        metric_fn = MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='micro'
        )
        super().__init__(metric_fn=metric_fn, name="Accuracy", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)


@define_metric
class Sensitivity(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 2, ignore_index: int = -100) -> None:
        # Sensitivity is the Recall of the positive class
        metric_fn = MulticlassRecall(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='none'
        )
        super().__init__(metric_fn=metric_fn, name="Sensitivity", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)

    def compute(self) -> torch.Tensor:
        # Isolate and return the score for class index 1 (positive class)
        return self.metric_fn.compute()[1]


@define_metric
class Specificity(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 2, ignore_index: int = -100) -> None:
        # Specificity is the True Negative Rate
        metric_fn = MulticlassSpecificity(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='none'
        )
        super().__init__(metric_fn=metric_fn, name="Specificity", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)

    def compute(self) -> torch.Tensor:
        # Return specificity for the positive class context
        return self.metric_fn.compute()[1]


@define_metric
class PosF1Score(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 2, ignore_index: int = -100) -> None:
        metric_fn = MulticlassF1Score(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='none'
        )
        super().__init__(metric_fn=metric_fn, name="Pos-F1", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)

    def compute(self) -> torch.Tensor:
        # Return F1 score exclusively for the positive class
        return self.metric_fn.compute()[1]


@define_metric
class AUROC(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 2, ignore_index: int = -100) -> None:
        metric_fn = MulticlassAUROC(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro'
        )
        super().__init__(metric_fn=metric_fn, name="AUROC", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)


@define_metric
class AUPRC(BaseMetric):
    def __init__(self, stage: str, num_classes: int = 2, ignore_index: int = -100) -> None:
        metric_fn = MulticlassAveragePrecision(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro'
        )
        super().__init__(metric_fn=metric_fn, name="AUPRC", stage=stage)

    def step(self, output: dict[str, torch.Tensor]) -> None:
        target_indices = torch.argmax(output["labels"], dim=-1)
        self.metric_fn.update(output["logits"], target_indices)
