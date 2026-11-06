from typing import Any, Callable, Generic, TypeVar, overload

import torch
from torch.utils.data import Dataset

from ..base import Base


T = TypeVar("T", bound="BaseDataset")


class DatasetBlueprint(Generic[T]):
    __slots__ = ("_cls", "_kwargs")

    def __init__(self, cls: type[T], **kwargs: Any) -> None:
        self._cls = cls
        self._kwargs = kwargs

    def __call__(self, mode: str, **runtime_kwargs) -> T:
        # Merge init kwargs with runtime kwargs (runtime wins)
        final_kwargs = {**self._kwargs, **runtime_kwargs}
        return self._cls(mode=mode, **final_kwargs)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self._kwargs.items())
        return f"<Blueprint: {self._cls.__name__}({params})>"


# --- Decorator Overloads ---
# Case 1: Called as @define_dataset (no parentheses)
@overload
def define_dataset(cls: type[T]) -> Callable[..., DatasetBlueprint[T]]: ...


# Case 2: Called as @define_dataset() (with parentheses/factory)
@overload
def define_dataset(
    cls: None = None,
) -> Callable[[type[T]], Callable[..., DatasetBlueprint[T]]]: ...


def define_dataset(
    cls: type[T] | None = None,
) -> (
    Callable[..., DatasetBlueprint[T]]
    | Callable[[type[T]], Callable[..., DatasetBlueprint[T]]]
):
    def create_factory(target_cls: type[T]) -> Callable[..., DatasetBlueprint[T]]:
        def wrapper(**kwargs: Any) -> DatasetBlueprint[T]:
            return DatasetBlueprint(target_cls, **kwargs)

        return wrapper

    if cls is not None:
        return create_factory(cls)

    def decorator(inner_cls: type[T]) -> Callable[..., DatasetBlueprint[T]]:
        return create_factory(inner_cls)

    return decorator


class BaseDataset(Base, Dataset):
    def __init__(self, mode: str, **kwargs: Any):
        Base.__init__(self)
        Dataset.__init__(self)
        self.mode = mode
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class BaseCollator(Base):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class BaseData(Base):
    def __init__(self, dataset: DatasetBlueprint[Any], collator: BaseCollator):
        super().__init__()
        self.dataset = dataset
        self.collator = collator

        self.train_ds: BaseDataset | None = None
        self.val_ds: BaseDataset | None = None
        self.test_ds: BaseDataset | None = None

    def setup(self, **kwargs) -> None:
        self.train_ds = self.dataset(mode="train", **kwargs)
        self.val_ds = self.dataset(mode="val", **kwargs)
        self.test_ds = self.dataset(mode="test", **kwargs)

    @staticmethod
    def _get_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(x, int):
            return x, x
        return x
