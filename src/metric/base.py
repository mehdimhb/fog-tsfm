import torch
import torch.nn as nn
from torchmetrics import Metric as TorchMetric
from typing import Callable, TypeVar, Generic

from ..base import Base


T = TypeVar("T", bound="MetricModule")


class MetricBlueprint(Base, Generic[T]):
    def __init__(self, cls: type[T], **kwargs) -> None:
        Base.__init__(self)
        self._cls = cls
        self.capture_config({"class": cls.__name__, "module": cls.__module__})
        del self._config["parameters"]["cls"]

    def __call__(self, stage: str) -> T:
        return self._cls(stage=stage, **self.config["parameters"])

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v}" for k, v in self.config["parameters"].items() if k != "metrics"
        )
        return f"<Blueprint: {self._cls.__name__}({params})>"


def define_metric(
    cls: type[T] | None = None,
) -> (
    Callable[..., MetricBlueprint[T]]
    | Callable[[type[T]], Callable[..., MetricBlueprint[T]]]
):

    def create_factory(target_cls: type[T]) -> Callable[..., MetricBlueprint[T]]:
        def wrapper(**kwargs) -> MetricBlueprint[T]:
            return MetricBlueprint(target_cls, **kwargs)

        return wrapper

    if cls is not None:
        return create_factory(cls)

    def decorator(inner_cls: type[T]) -> Callable[..., MetricBlueprint[T]]:
        return create_factory(inner_cls)

    return decorator


def define_metric_set(cls: type[T] | None = None):

    def create_factory(target_cls: type[T]):
        def wrapper(
            train_metrics: list[MetricBlueprint],
            eval_metrics: list[MetricBlueprint]
        ) -> MetricBlueprint[T]:
            return MetricBlueprint(
                target_cls, train_metrics=train_metrics, eval_metrics=eval_metrics
            )

        return wrapper

    if cls is not None:
        return create_factory(cls)

    def decorator(inner_cls: type[T]):
        return create_factory(inner_cls)

    return decorator


class MetricModule(nn.Module):
    def __init__(self):
        super().__init__()

    def step(self, output: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def compute_and_reset(self) -> dict[str, float]:
        raise NotImplementedError


class BaseMetric(MetricModule):
    def __init__(self, metric_fn: TorchMetric, name: str, stage: str) -> None:
        super().__init__()
        self.metric_fn = metric_fn
        self.name = name
        self.stage = stage

    def __repr__(self) -> str:
        return f"{self.stage}/metric/{self.name}"

    def step(self, output: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        """Default compute behavior. Subclasses can override this."""
        return self.metric_fn.compute()

    def compute_and_reset(self) -> dict[str, float]:
        # Call self.compute() so that child classes like Sensitivity
        # and PosF1Score can override it with custom logic!
        value = self.compute().item()
        self.metric_fn.reset()
        return {str(self): value}


@define_metric_set
class MetricSet(MetricModule):
    def __init__(
        self,
        train_metrics: list[MetricBlueprint],
        eval_metrics: list[MetricBlueprint],
        stage: str
    ) -> None:
        super().__init__()
        self.stage = stage
        self.train_metrics = nn.ModuleList([blueprint(stage=stage) for blueprint in train_metrics])
        self.eval_metrics = nn.ModuleList([blueprint(stage=stage) for blueprint in eval_metrics])

    def step(self, output: dict[str, torch.Tensor]) -> None:
        if self.stage == 'train':
            for metric in self.train_metrics:
                metric.step(output)
        else:
            for metric in self.eval_metrics:
                metric.step(output)

    def compute_and_reset(self) -> dict[str, float]:
        results = {}
        if self.stage == 'train':
            for metric in self.train_metrics:
                results.update(metric.compute_and_reset())
        else:
            for metric in self.eval_metrics:
                results.update(metric.compute_and_reset())
        return results
