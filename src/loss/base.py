import torch
import torch.nn as nn
from torchmetrics import Metric as TorchMetric
from torchmetrics import MeanMetric as TorchMeanMetric
from typing import Callable, TypeVar, Generic

from ..base import Base


T = TypeVar("T", bound="BaseLoss")


class LossBlueprint(Base, Generic[T]):
    def __init__(self, cls: type[T], **kwargs) -> None:
        Base.__init__(self)
        self._cls = cls
        self.capture_config({"class": cls.__name__, "module": cls.__module__})
        del self._config["parameters"]["cls"]

    def __call__(self, stage: str) -> T:
        return self._cls(stage=stage, **self.config["parameters"])

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v}" for k, v in self.config["parameters"].items() if k != "losses"
        )
        return f"<Blueprint: {self._cls.__name__}({params})>"


def define_loss(
    cls: type[T] | None = None,
) -> (
    Callable[..., LossBlueprint[T]]
    | Callable[[type[T]], Callable[..., LossBlueprint[T]]]
):

    def create_factory(target_cls: type[T]) -> Callable[..., LossBlueprint[T]]:
        def wrapper(**kwargs) -> LossBlueprint[T]:
            return LossBlueprint(target_cls, **kwargs)

        return wrapper

    if cls is not None:
        return create_factory(cls)

    def decorator(inner_cls: type[T]) -> Callable[..., LossBlueprint[T]]:
        return create_factory(inner_cls)

    return decorator


def define_composite_loss(cls: type[T] | None = None):
    def create_factory(target_cls: type[T]):
        def wrapper(
            losses: list[LossBlueprint], weights: list[float] | None = None
        ) -> LossBlueprint[T]:
            return LossBlueprint(target_cls, losses=losses, weights=weights)

        return wrapper

    if cls is not None:
        return create_factory(cls)

    def decorator(inner_cls: type[T]):
        return create_factory(inner_cls)

    return decorator


class BaseLoss(nn.Module):
    def __init__(self, loss_fn: TorchMetric, name: str, stage: str) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.name = name
        self.stage = stage

    def __repr__(self) -> str:
        return f"{self.stage}/loss/{self.name}"

    def step(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def compute_and_reset(self) -> dict[str, float]:
        value = self.loss_fn.compute().item()
        self.loss_fn.reset()
        return {str(self): value}


@define_composite_loss
class CompositeLoss(BaseLoss):
    def __init__(
        self,
        losses: list[LossBlueprint],
        stage: str,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__(loss_fn=TorchMeanMetric(), name="Total", stage=stage)
        self.sub_losses = nn.ModuleList(
            [blueprint(stage=stage) for blueprint in losses]
        )
        self.weights = weights or [1.0] * len(losses)

        if len(self.weights) != len(self.sub_losses):
            raise ValueError(
                f"Weights count ({len(self.weights)}) must match losses count ({len(self.sub_losses)})"
            )

        self.register_buffer(
            "_weights_tensor", torch.tensor(self.weights, dtype=torch.float32)
        )

    def step(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self._weights_tensor.device)

        for i, sub_loss in enumerate(self.sub_losses):
            val = sub_loss.step(output)
            total_loss += self._weights_tensor[i] * val

        self.loss_fn.update(total_loss)
        return total_loss

    def compute_and_reset(self) -> dict[str, float]:
        results = {}

        for sub_loss in self.sub_losses:
            results.update(sub_loss.compute_and_reset())

        total_value = self.loss_fn.compute().item()
        self.loss_fn.reset()
        results.update({str(self): total_value})

        return results
