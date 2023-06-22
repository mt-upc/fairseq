from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.modules import LayerNorm, FairseqDropout
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase

    
@dataclass
class ContextEncoderConfig(TransformerConfig):
    dropout: float = field(
        default=0.0,
        metadata={"help": "context encoder dropout"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "context encoder activation dropout"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "context encoder attention dropout"}
    )
    freeze: bool = field(
        default=False,
        metadata={"help": "freeze context encoder"}
    )

class ContextEncoder(nn.Module):
    def __init__(self, cfg: ContextEncoderConfig):
        super().__init__()

        self.cfg = cfg
        self.dropout_module = FairseqDropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(cfg.encoder.layers)]
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.layer_norm = LayerNorm(cfg.encoder.embed_dim)

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        if not self.normalize_before:
            x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x, padding_mask)

        if self.normalize_before:
            x = self.layer_norm(x)

        return x
