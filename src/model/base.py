import json
from pathlib import Path

import torch
import torch.nn as nn

from ..base import Base, BaseConfig


class BaseModel(Base, nn.Module):
    def __init__(self):
        Base.__init__(self)
        nn.Module.__init__(self)

    @staticmethod
    def _get_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(x, int):
            return x, x
        return x

    @classmethod
    def load(cls, config_path: str | Path, weight_path: str | Path):
        config = json.loads(Path(config_path).read_text())
        params = cls._load_params(config["parameters"])
        # del params['encoder_layer']
        instance = cls(**params)
        ckpt = torch.load(weight_path)
        state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
        instance.load_state_dict(state_dict, strict=False)
        return instance

    def load_wieght(self, weight_path: str | Path):
        ckpt = torch.load(weight_path)
        state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
        self.load_state_dict(state_dict, strict=False)
        return self


class BaseLayerConfig(BaseConfig):
    pass
