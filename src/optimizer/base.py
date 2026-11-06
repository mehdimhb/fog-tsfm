import torch.nn as nn
from lightning.pytorch import LightningModule

from ..base import Base


class BaseOptimizer(Base):
    def setup(self, model: LightningModule, total_steps: int) -> dict:
        raise NotImplementedError

    @staticmethod
    def create_weight_decay_groups(
        model: nn.Module,
        weight_decay: float,
        whitelist_modules: tuple[type[nn.Module], ...],
        blacklist_names: tuple[str, ...],
    ):
        decay = set()
        no_decay = set()

        for mn, m in model.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if any(name in fpn.lower() for name in blacklist_names) or any(
                    name in m.__class__.__name__.lower() for name in blacklist_names
                ):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_modules):
                    decay.add(fpn)

        # Validate coverage
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, (
            f"Parameters {inter_params} made it into both decay/no_decay sets!"
        )

        assert len(param_dict.keys() - union_params) == 0, (
            f"Parameters {param_dict.keys() - union_params} were not separated found"
        )

        return [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
