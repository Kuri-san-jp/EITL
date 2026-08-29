"""An **exact replica** of SongBench's reward `Generator`.

The module structure, ordering, and names match `Generator` in
external/SongBench/model.py exactly (otherwise the safetensors state_dict keys
would not line up and load_state_dict would break).
Instantiating hydra `_target_: model.Generator` requires cwd=repo and importing
`model`, which is impractical for in-process use, so we reproduce the identical
definition here and construct it directly.

There must be no divergence from the source (external/SongBench/model.py). The
layer definitions and the forward equation (`x = self.proj(x) * 4.5 + 5.5`,
range [1, 10]) are identical as well.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_features, ffd_hidden_size, num_classes, attn_layer_num):
        super(Generator, self).__init__()
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=in_features,
                    num_heads=8,
                    dropout=0.2,
                    batch_first=True,
                )
                for _ in range(attn_layer_num)
            ]
        )
        self.ffd = nn.Sequential(
            nn.Linear(in_features, ffd_hidden_size),
            nn.ReLU(),
            nn.Linear(ffd_hidden_size, in_features),
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features * 2, num_classes)
        self.proj = nn.Tanh()

    def forward(self, ssl_feature, judge_id=None):
        """ssl_feature: [B, T, D] -> output: [B, num_classes]"""
        B, T, D = ssl_feature.shape
        ssl_feature = self.ffd(ssl_feature)
        tmp_ssl_feature = ssl_feature
        for attn in self.attn:
            tmp_ssl_feature, _ = attn(tmp_ssl_feature, tmp_ssl_feature, tmp_ssl_feature)
        ssl_feature = self.dropout(
            torch.concat(
                [torch.mean(tmp_ssl_feature, dim=1), torch.max(ssl_feature, dim=1)[0]],
                dim=1,
            )
        )  # B, 2D
        x = self.fc(ssl_feature)
        x = self.proj(x) * 4.5 + 5.5
        return x


def build_generator(in_features: int, ffd_hidden_size: int,
                    num_classes: int, attn_layer_num: int) -> Generator:
    return Generator(in_features=in_features, ffd_hidden_size=ffd_hidden_size,
                     num_classes=num_classes, attn_layer_num=attn_layer_num)
