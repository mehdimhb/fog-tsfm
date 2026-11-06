from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class MultiBiasEncoder(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.scale_weighted_sum = config.scale_weighted_sum
        self.loc_weighted_sum = config.loc_weighted_sum
        self.use_loc_emb = config.use_loc_emb
        self.out_channels = config.mb_emb_dim

        ws = nn.Parameter(torch.empty(config.num_bias, self.out_channels))
        bs = nn.Parameter(torch.empty(config.num_bias, self.out_channels))
        nn.init.uniform_(ws, -1, 1)
        nn.init.uniform_(bs, -1, 1)
        if config.multibias_trainable:
            self.ws = ws
            self.bs = bs
        else:
            self.register_buffer("ws", ws)
            self.register_buffer("bs", bs)

        scales = (
            [config.base_bias**i for i in range(-(config.num_bias - 1) // 2, (config.num_bias + 1) // 2)]
            if config.scales is None
            else config.scales
        )
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        locs = [0] if config.locs is None else config.locs
        self.register_buffer("locs", torch.tensor(locs, dtype=torch.float32))

        if self.use_loc_emb:
            self.loc_emb = nn.Parameter(torch.empty(self.locs.shape[-1], self.out_channels))
            nn.init.trunc_normal_(self.loc_emb, std=0.02)
        self.ln = nn.LayerNorm(self.out_channels, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_x = x
        if enc_x.ndim > 2:
            enc_x = enc_x.view(enc_x.shape[0], -1)

        eps = 1e-6
        loc_dis = enc_x[..., None] - self.locs
        if self.loc_weighted_sum:
            loc_w = 1 / (torch.log(torch.abs(loc_dis) + 1) + eps)
            loc_w = loc_w / loc_w.sum(dim=-1, keepdim=True)
        if self.scale_weighted_sum:
            scale_w = 1 / (torch.abs(torch.log(torch.abs(loc_dis[..., None]) / self.scales + eps)) + eps)
            scale_w = scale_w / scale_w.sum(dim=-1, keepdim=True)

        enc_x = loc_dis
        enc_x = enc_x[..., None, None] * self.ws + self.bs * self.scales.unsqueeze(dim=-1)
        enc_x = self.ln(enc_x)

        if self.scale_weighted_sum:
            enc_x = enc_x * scale_w[..., None]
        enc_x = enc_x.sum(dim=-2)

        if self.use_loc_emb:
            enc_x = enc_x + self.loc_emb

        if self.loc_weighted_sum:
            enc_x = enc_x * loc_w[..., None]
        return enc_x.sum(dim=-2)


class WindowNormEncoder(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        assert config.model_series_size % config.window_size == 0
        self.window_size = config.window_size
        self.stride = config.stride
        self.window_num = (config.model_series_size - config.window_size) // config.stride + 1
        self.in_channels = config.num_channels
        self.wne_use_in = config.wne_use_in
        self.wne_use_instance_stats = config.wne_use_instance_stats
        self.use_ms_embed = config.use_ms_embed
        self.mean_std_encode = config.mean_std_encode

        embed_dim = config.window_emb_dim
        self.window_embedding = nn.Conv1d(1, embed_dim, kernel_size=self.window_size, stride=self.window_size)
        self.window_embed_norm = nn.LayerNorm(embed_dim)

        if self.use_ms_embed:
            if config.mean_std_encode != "mbe":
                raise ValueError("This handoff supports the official checkpoint_bias9 MultiBias path only.")
            self.multi_bias_encoder = MultiBiasEncoder(config)
            self.ms_embed_fc = nn.Sequential(
                nn.Linear(embed_dim + 2 * config.mb_emb_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        self.use_channel_emb_sum = config.use_channel_emb_sum
        if config.num_channels > 1:
            self.channel_embedding = nn.Sequential(
                nn.Linear(config.num_channels * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.channel_embedding = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        num_windows, window_size = self.window_num, self.window_size

        x_in_mean = x.mean(dim=-1, keepdim=True).detach().expand(batch_size, channels, num_windows).unsqueeze(-1)
        x_in_std = x.std(dim=-1, keepdim=True).detach().expand(batch_size, channels, num_windows).unsqueeze(-1)

        x = x.view(batch_size, channels, -1, window_size)
        x_wn_mean = x.mean(dim=-1, keepdim=True).detach()
        x_wn_std = x.std(dim=-1, keepdim=True).detach()

        if self.wne_use_instance_stats:
            x_mean = x_in_mean
            x_std = x_in_std
        else:
            x_mean = x_wn_mean
            x_std = x_wn_std

        if self.wne_use_in:
            x = (x - x_in_mean) / (x_in_std + 1e-8)
        else:
            x = (x - x_wn_mean) / (x_wn_std + 1e-8)

        x = x.view(-1, 1, length)
        x_embed = self.window_embedding(x).transpose(1, 2).view(batch_size, channels, num_windows, -1)
        x_embed = self.window_embed_norm(x_embed)

        if self.use_ms_embed:
            x_mean = x_mean.view(batch_size, channels, num_windows)
            x_std = x_std.view(batch_size, channels, num_windows)
            mean_embeds = self.multi_bias_encoder(x_mean).view(batch_size, channels, num_windows, -1)
            std_embeds = self.multi_bias_encoder(x_std).view(batch_size, channels, num_windows, -1)
            xms_embed = torch.cat((x_embed, mean_embeds, std_embeds), dim=-1)
            x_embed = self.ms_embed_fc(xms_embed)

        if self.use_channel_emb_sum:
            x_embed = x_embed.sum(dim=1)
            x_embed = self.window_embed_norm(x_embed).transpose(1, 2)
        else:
            x_embed = x_embed.transpose(1, 2).contiguous().view(batch_size, num_windows, -1)
            x_embed = self.channel_embedding(x_embed).transpose(1, 2)

        return x_embed


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % heads == 0
        inner_dim = head_dim * heads
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv = self.to_qkv(x).reshape(batch_size, num_tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(self.attend(dots))
        output = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, num_tokens, self.heads * self.head_dim)
        return self.to_out(output)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.extend(
                [
                    PreNorm(dim, Attention(dim, heads, head_dim, dropout, attn_dropout)),
                    nn.Identity(),
                    nn.Identity(),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    nn.Identity(),
                    nn.Identity(),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers) // 6):
            attn, ls1, dp1, ff, ls2, dp2 = self.layers[i * 6 : i * 6 + 6]
            x = dp1(ls1(attn(x))) + x
            x = dp2(ls2(ff(x))) + x
        return x

class WinT(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        in_channels = 128
        self.features = None
        self.pool_mode = "cls"
        self.num_windows = 31
        self.transformer_mask_type = config.transformer_mask_type
        self.mask_token_num = int(config.transformer_mask_scale * self.num_windows)
        self.cls_token = nn.Parameter(torch.empty(1, 1, in_channels))
        nn.init.normal_(self.cls_token, std=1e-6)

        position = torch.arange(config.transformer_max_tokens).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2).float() * (-math.log(10000.0) / in_channels))
        pos_embed = torch.zeros(1, config.transformer_max_tokens, in_channels)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_embed", pos_embed)
        self.pos_drop = nn.Dropout(config.pos_dropout)

        self.transformer = Transformer(
            in_channels,
            config.transformer_depth,
            config.transformer_heads,
            config.transformer_head_dim,
            config.transformer_mlp_dim,
            config.dropout,
            config.attn_dropout,
        )
        self.norm = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, config.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_embed = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_embed = torch.cat((cls_tokens, x_embed), dim=1)
        x_embed = x_embed + self.pos_embed[:, : x_embed.shape[1], :]
        x_embed = self.pos_drop(x_embed)
        output = self.transformer(x_embed)
        output_norm = self.norm(output)
        self.features = output_norm[:, 0]
        return self.fc(self.features)


class NuTimeChannelEncoder(nn.Module):
    """Frozen NuTime channel encoder.

    Input:  windows with shape (B, C, L) where L >= 496 (e.g., 512)
    Output: per-channel CLS features with shape (B, C, 128)
    """

    def __init__(self, backbone: nn.Sequential) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, L), got {tuple(windows.shape)}")

        batch_size, channels, length = windows.shape

        if length < 496:
            raise ValueError(f"Input length must be at least 496 timesteps, got {length}")

        # Slice the first 496 timesteps and collapse Batch and Channel dimensions
        x = windows[:, :, :496].reshape(batch_size * channels, 1, 496)

        with torch.no_grad():
            _ = self.backbone(x)
            cls = self.backbone[1].features

        # Dynamically reshape the output using the actual channel count (e.g., 24)
        return cls.reshape(batch_size, channels, 128).detach()


def _nutime_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_channels=1,
        model_series_size=496,
        window_size=16,
        stride=16,
        window_emb_dim=128,
        use_ms_embed=True,
        mean_std_encode="mbe",
        mb_emb_dim=32,
        num_bias=9,
        base_bias=10,
        scales=None,
        locs=None,
        multibias_trainable=True,
        scale_weighted_sum=True,
        loc_weighted_sum=False,
        use_loc_emb=False,
        use_channel_emb_sum=False,
        wne_use_in=False,
        wne_use_instance_stats=False,
        transformer_depth=6,
        transformer_heads=8,
        transformer_head_dim=64,
        transformer_mlp_dim=512,
        transformer_max_tokens=1024,
        transformer_mask_type="none",
        transformer_mask_scale=0.0,
        dropout=0.1,
        pos_dropout=0.0,
        attn_dropout=0.1,
        out_dim=256,
    )


def get_loaded_nutime_channel_encoder(
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> NuTimeChannelEncoder:
    """Load official NuTime checkpoint_bias9 and return a frozen (B,9,500)->(B,9,128) encoder."""

    if weights_path is None:
        weights_path = Path(__file__).resolve().with_name("checkpoint_bias9.pth")
    weights_path = Path(weights_path)
    config = _nutime_config()
    encoder = WindowNormEncoder(config)
    transformer = WinT(config)
    backbone = nn.Sequential(encoder, transformer)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state = {
        key.replace("backbone.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }

    model_state = backbone.state_dict()
    compatible = {
        key: value
        for key, value in clean_state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    result = backbone.load_state_dict(compatible, strict=False)

    allowed_missing = {"1.fc.weight", "1.fc.bias"}
    missing = sorted(set(result.missing_keys) - allowed_missing)
    unexpected = sorted(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected NuTime checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if len(compatible) != 82:
        raise RuntimeError(f"Expected 82 compatible NuTime tensors, loaded {len(compatible)}")

    model = NuTimeChannelEncoder(backbone).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"NuTime encoder is not frozen; trainable params={trainable}")
    return model
