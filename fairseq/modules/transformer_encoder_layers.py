from typing import Optional

import torch
import torch.nn as nn

from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.modules import LayerNorm
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase


class TransformerEncoderLayers(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(cfg.encoder.layers)]
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.layer_norm = LayerNorm(cfg.encoder.embed_dim)

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        
        if not self.normalize_before:
            x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        if self.normalize_before:
            x = self.layer_norm(x)
        return x