import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .networks import TokenGeneratorUnit, InceptionModule, TransformerEnc


class UniShapeModel(nn.Module):
    def __init__(self,
                 series_size: int = 512,
                 in_channels: int = 128,
                 out_channels: int = 8,
                 depth: int = 6,
                 heads: int = 8,
                 mlp_dim: int = 512,
                 pool_mode: str = 'cls',
                 pre_training: bool = False,
                 shape_ratio: float = 0.3,
                 proto_tau: float = 0.5,
                 shape_alpha: float = 0.01,
                 shape_sparse_ratio: float = 0.4,
                 scale_len: int = 4,
                 return_all_cls_tokens: bool = False): # <-- Added parameter

        super(UniShapeModel, self).__init__()

        self.return_all_cls_tokens = return_all_cls_tokens
        self.shape_sparse_ratio = shape_sparse_ratio
        self.shape_alpha = shape_alpha
        self.proto_tau = proto_tau
        self.scale_len = scale_len
        self.encoder_scale_list = nn.ModuleList()
        self.shape_ratio = shape_ratio

        self.act_gelu_inc = nn.GELU()
        self.layer_norm_inc = nn.LayerNorm(in_channels)
        self.inceptime_token = nn.Sequential(InceptionModule(128, 32), InceptionModule(128, 32))
        self.drop_token = nn.Dropout(p=0.15)

        window_size_list = [64, 32, 16, 8]
        last_stride = None

        # Replace config logic with local variables
        for current_window_size in window_size_list:

            if series_size < 64:
                current_stride = 2
            else:
                current_stride = 4

            if pre_training:
                current_stride = current_window_size

            num_patches = int((series_size - current_window_size) / current_stride + 1)

            tokgen_unit = TokenGeneratorUnit(hidden_dim=128,
                                             num_patches=num_patches,
                                             patch_window_size=current_window_size,
                                             scalar_scales=None,
                                             hidden_dim_scalar_enc=32,
                                             epsilon_scalar_enc=1.1,
                                             patch_len=current_window_size,
                                             stide_len=current_stride)

            self.encoder_scale_list.append(tokgen_unit)
            last_stride = current_stride

        self.num_windows = int((series_size - window_size_list[-1]) / last_stride) + 1
        num_tokens = self.num_windows + 1 if pool_mode == 'cls' else self.num_windows

        self.pre_training = pre_training

        self.vit_unit = TransformerEnc(hidden_dim=in_channels, num_patches=num_tokens, depth=depth,
                                       heads=heads, mlp_dim=mlp_dim, dim_head=64,
                                       dropout=0.1, device=None)

        self.act_gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(in_channels * 1)

        if self.pre_training:
            self.fc = nn.Linear(in_channels * 1, out_channels)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, out_channels)
            )

        self.fc_token_shape = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, in_channels)
        )

        self.class_proto_centers = nn.Parameter(torch.zeros(out_channels, in_channels))
        self.attention_head = nn.Sequential(
            nn.Linear(128, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels=None):
        B, C, L = x.shape
        N = self.num_windows

        window_size_list = [64, 32, 16, 8]
        cls_tokens = None
        x_embed = None

        for _i in range(self.scale_len):
            _shape_size = window_size_list[_i]
            if (L - 4) <= _shape_size:
                continue

            x_embed = self.encoder_scale_list[_i](x)

            if cls_tokens is None:
                cls_incep_token_list = self.inceptime_token(x_embed.permute(0,2,1)).permute(0,2,1)
            else:
                input_x_embed = torch.cat((cls_tokens, x_embed), dim=1).permute(0,2,1)
                cls_incep_token_list = self.inceptime_token(input_x_embed).permute(0,2,1)

            cls_incep_token_list = self.drop_token(self.layer_norm_inc(cls_incep_token_list))
            cls_incep_token_list = self.act_gelu_inc(cls_incep_token_list)
            _attn_x_score = self.attention_head(cls_incep_token_list)
            attn_shape_embds = cls_incep_token_list * _attn_x_score

            cls_tokens = torch.mean(attn_shape_embds, dim=1).unsqueeze(1)

        x_embed = x_embed.squeeze(1)

        # Optionally retrieve intermediate layer cls tokens
        if self.return_all_cls_tokens:
            trans_enc_class_token, _, all_cls_tokens = self.vit_unit(
                x_embed, cls_token_in=cls_tokens, return_all_layers=True)
            # Output representation used for zero-shot tasks + sequence of intermediate cls tokens
            return trans_enc_class_token, all_cls_tokens
        else:
            trans_enc_class_token, _ = self.vit_unit(x_embed, cls_token_in=cls_tokens)
            # Output representation used for zero-shot tasks
            return trans_enc_class_token


def get_loaded_uni_shape(weights_path, return_all_cls_tokens=False) -> UniShapeModel: # <-- Added parameter
    model = UniShapeModel(
        series_size=512,       # Ensure your input data is interpolated to 512
        in_channels=128,       # Hidden dimension size from token generator
        out_channels=8,        # Classes for your dataset
        pre_training=False,    # Zero-shot evaluation setting
        return_all_cls_tokens=return_all_cls_tokens # <-- Pass parameter
    )

    # 3. Load the state dictionary
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    # Adjust if the state dict is nested inside a checkpoint dictionary
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # 4. Load the weights into the model
    # We use strict=False because the classification head parameters (self.fc)
    # between the pre-trained model and your initialized model might differ in size
    model.load_state_dict(state_dict, strict=False)

    return model
