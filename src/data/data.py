from torch.utils.data import default_collate

from .base import BaseData
from .features_dataset import FeatureDataset
from .collator import Collator


class FeatureData(BaseData):
    def __init__(
        self,
        lmdb_path: str = "datasets/new_lmdb_features",
        models: list[str] = None,
        num_classes: int = 2,
        combo: str = "p0123_s2",
        one_hot_labels: bool = False,
    ) -> None:
        self.lmdb_path = lmdb_path
        # Default to the primary models if none are provided
        self.models = models if models is not None else ['unishape', 'mantis', 'nutime']
        self.num_classes = num_classes
        self.one_hot_labels = one_hot_labels
        self.combo = combo

        super().__init__(
            dataset=FeatureDataset(
                lmdb_path=self.lmdb_path,
                models=self.models,
                num_classes=self.num_classes,
                one_hot_labels=self.one_hot_labels,
                combo=self.combo,
            ),
            # default_collate seamlessly stacks dictionaries of tensors into a single batched dictionary
            collator=default_collate,
        )

        self.capture_config()
