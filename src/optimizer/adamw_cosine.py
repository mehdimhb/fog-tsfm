import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from lightning.pytorch import LightningModule

from .base import BaseOptimizer

class AdamWLinearProbe(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        min_lr: float = 1e-6,
        warmup_steps: int = 500, # NEW: Number of steps for linear warmup
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.capture_config()

    def setup(self, model: LightningModule, total_steps: int) -> dict:
        optimizer = self._setup_optimizer(model)

        # Only setup scheduler if total_steps is provided
        if total_steps is not None and total_steps > 0:
            # Safety catch: don't let warmup exceed 20% of total steps
            warmup_iters = min(self.warmup_steps, max(1, total_steps // 5))

            # 1. Warmup Schedule (Linear increase from min_lr to learning_rate)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.min_lr / self.learning_rate,
                end_factor=1.0,
                total_iters=warmup_iters
            )

            # 2. Cosine Decay Schedule (Decay from learning_rate to min_lr)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_iters,
                eta_min=self.min_lr
            )

            # 3. Chain them together
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_iters]
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        return {"optimizer": optimizer}

    def _setup_optimizer(self, model: LightningModule) -> optim.Optimizer:
        # Separate parameters that should be decayed from those that shouldn't
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Do not decay biases or normalization scales
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999)
        )
        return optimizer
