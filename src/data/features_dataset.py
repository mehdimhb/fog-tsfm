import lmdb
import pickle
import torch
import torch.nn.functional as F

from .base import BaseDataset, define_dataset


class _FeatureDataset(BaseDataset):
    def __init__(
        self,
        mode: str,
        combo: str = "p0123_s2",
        lmdb_path: str = "datasets/new_lmdb_features",
        models: list[str] = None,
        num_classes: int = 2,
        one_hot_labels: bool = False,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            combo (str): The placement & sensor combination key (e.g., 'p03_s0', 'p0123_s2').
            lmdb_path (str): Path to the LMDB features directory.
            models (list[str], optional): List of feature representations to extract.
            num_classes (int): Number of target classes.
            one_hot_labels (bool): If True, returns one-hot float tensor; if False, integer class index.
        """
        if models is None:
            models = ['unishape', 'mantis', 'nutime']

        self.lmdb_path = lmdb_path
        self.mode = mode
        self.combo = combo
        self.models = models
        self.num_classes = num_classes
        self.one_hot_labels = one_hot_labels

        self.env = None
        self.txn = None

        # -------------------------------------------------------------
        # Read keys & inspect first sample to set self.num_channels
        # -------------------------------------------------------------
        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:
            keys_data = txn.get(f"{self.mode}_keys".encode('ascii'))
            if keys_data is None:
                raise ValueError(f"Key list '{self.mode}_keys' not found in LMDB at {self.lmdb_path}.")
            self.keys = pickle.loads(keys_data)

            # Inspect the first sample to dynamically infer self.num_channels
            first_val = txn.get(self.keys[0].encode('ascii'))
            sample_dict = pickle.loads(bytes(first_val))

            if self.combo not in sample_dict:
                raise KeyError(
                    f"Combination '{self.combo}' not found in LMDB. "
                    f"Available sample keys include: {[k for k in sample_dict.keys() if k != 'label']}"
                )

            combo_features = sample_dict[self.combo]

            # Find the first available model feature array to extract channel count (shape: (C, D))
            self.num_channels = None
            for model_name in self.models:
                feat_key = f"{model_name}_features"
                if feat_key in combo_features:
                    self.num_channels = combo_features[feat_key].shape[0]
                    break

            if self.num_channels is None:
                raise ValueError(f"None of the requested models {self.models} exist in combo '{self.combo}'.")

        temp_env.close()

    def _init_db(self):
        """Lazy initialization of LMDB environment for DataLoader worker processes."""
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.txn = self.env.begin(buffers=True)

    def __len__(self) -> int:
        return len(self.keys)

    def _get_single_sample(self, idx: int) -> tuple:
        key = self.keys[idx].encode('ascii')
        value = self.txn.get(key)
        data_dict = pickle.loads(bytes(value))

        combo_dict = data_dict.get(self.combo, {})

        # Load requested model features for the specified combination
        features = {}
        for model_name in self.models:
            feature_key = f"{model_name}_features"
            if feature_key in combo_dict:
                features[model_name] = torch.from_numpy(combo_dict[feature_key]).float()

        raw_label = data_dict['label']
        label_tensor = torch.tensor(raw_label, dtype=torch.long)

        if self.one_hot_labels:
            label_output = F.one_hot(label_tensor, num_classes=self.num_classes).float()
        else:
            label_output = label_tensor

        return features, label_output

    def __getitem__(self, idx: int) -> dict:
        if self.env is None:
            self._init_db()

        features, label = self._get_single_sample(idx)

        output = {}
        for model_name in self.models:
            if model_name in features:
                output[f'{model_name}_features'] = features[model_name]

        output['labels'] = label
        return output


FeatureDataset = define_dataset(_FeatureDataset)