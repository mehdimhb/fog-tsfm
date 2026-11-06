import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base import BaseModel

class ChannelAttentionPooler(nn.Module):
    """Dynamically weights channels before pooling."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class LightweightCrossAttention(nn.Module):
    """Cross-attention where the sequence length of Keys/Values can vary."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, query_seq: torch.Tensor, kv_seq: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=query_seq,
            key=kv_seq,
            value=kv_seq,
            key_padding_mask=key_padding_mask
        )
        x = self.norm(query_seq + self.dropout(attn_out))
        out = self.proj(x)
        return self.norm_out(x + self.dropout(out))


class LightweightSelfAttention(nn.Module):
    """Self-attention explicitly designed for single-modality processing."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        x = self.norm(x + self.dropout(attn_out))
        out = self.proj(x)
        return self.norm_out(x + self.dropout(out))


class Ensemble(BaseModel):
    def __init__(
        self,
        active_modalities: List[str] = ['unishape', 'mantis', 'nutime'],
        num_channels: int = 9,
        dim: int = 128,
        num_classes: int = 8,
        num_heads: int = 4,
        dropout: float = 0.3,
        token_mask_prob: float = 0.15
    ):
        super().__init__()
        self.capture_config()
        self.token_mask_prob = token_mask_prob
        self.active_modalities = active_modalities

        # Native feature dimension of each foundation model
        self.seq_dims = {
            'unishape': 128,
            'mantis': 256,
            'nutime': 128,
        }

        self.active_seq = [m for m in active_modalities if m in self.seq_dims]

        if not self.active_seq:
            raise ValueError("At least one valid modality must be active.")

        # Determine if we need self-attention (1 modality) or cross-attention (>1 modalities)
        self.is_single_seq = len(self.active_seq) == 1

        # --- 1. Per-Modality Projection, Attention and Pooling ---
        self.seq_projs = nn.ModuleDict()
        self.seq_embeds = nn.ParameterDict()
        self.attns = nn.ModuleDict()  # Renamed from cross_attns to handle both types
        self.poolers = nn.ModuleDict()

        for mod in self.active_seq:
            # Projections
            self.seq_projs[mod] = nn.Sequential(
                nn.Linear(self.seq_dims[mod], dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout)
            )
            # Spatial/Channel Embeddings
            self.seq_embeds[mod] = nn.Parameter(torch.zeros(1, num_channels, dim))

            # Conditionally load Self or Cross attention block
            if self.is_single_seq:
                self.attns[mod] = LightweightSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
            else:
                self.attns[mod] = LightweightCrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)

            # Poolers
            self.poolers[mod] = ChannelAttentionPooler(dim)

        # --- 2. Classification Head ---
        # Each active modality contributes `dim` to the concatenated vector
        head_in_dim = dim * len(self.active_seq)

        self.head = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for mod in self.active_seq:
            nn.init.trunc_normal_(self.seq_embeds[mod], std=0.02)

    def _generate_token_mask(self, batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.training or self.token_mask_prob <= 0.0:
            return None

        rand_tensor = torch.rand(batch_size, seq_len, device=device)
        mask = rand_tensor < self.token_mask_prob

        # Check for sequences where every token is masked
        all_masked = mask.all(dim=-1)

        if all_masked.any():
            # Pick a random token index to unmask for every item in the batch
            random_unmask = torch.randint(0, seq_len, (batch_size,), device=device)
            # Apply the unmasking only to the rows that were entirely masked
            mask[all_masked, random_unmask[all_masked]] = False

        return mask

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = next(iter(batch.values())).size(0)
        fused_features = []

        # --- Phase 1: Project and embed all active features ---
        seq_x = {}
        for mod in self.active_seq:
            raw_feat = batch[f'{mod}_features']
            seq_x[mod] = self.seq_projs[mod](raw_feat) + self.seq_embeds[mod]

        # --- Phase 2: Attention and Pooling ---
        for mod in self.active_seq:

            if self.is_single_seq:
                # --- Single Modality: Self Attention Path ---
                seq = seq_x[mod]
                seq_len = seq.size(1)
                mask = self._generate_token_mask(batch_size, seq_len, seq.device)

                fused = self.attns[mod](seq, key_padding_mask=mask)
            else:
                # --- Multiple Modalities: Cross Attention Path ---
                other_xs = [seq_x[other] for other in self.active_seq if other != mod]
                kv_seq = torch.cat(other_xs, dim=1)
                seq_len = kv_seq.size(1)
                mask = self._generate_token_mask(batch_size, seq_len, kv_seq.device)

                fused = self.attns[mod](seq_x[mod], kv_seq, key_padding_mask=mask)

            pooled = self.poolers[mod](fused)
            fused_features.append(pooled)

        # --- Phase 3: Final Fusion and Classification ---
        fused_backbone = torch.cat(fused_features, dim=-1)

        logits = self.head(fused_backbone)

        return {
            'logits': logits,
            'labels': batch.get('labels')
        }
