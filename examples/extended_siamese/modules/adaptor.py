from functools import partial
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch import nn

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout


@dataclass
class AdaptorConfig(FairseqDataclass):
    embed_dim: int = field(
        default=1024,
        metadata={"help": "dimension of the Adaptor"}
    )
    num_layers: int = field(
        default=0,
        metadata={"help": "# of Adaptor layers"}
    )
    kernel_size: int = field(
        default=3,
        metadata={"help": "kernel size of each Conv1d in the Adaptor"}
    )
    stride: int = field(
        default=2,
        metadata={"help": "stride of each Conv1d in the Adaptor"}
    )
    pre_projection: bool = field(
        default=False,
        metadata={"help": "whether to apply a projection before the Adaptor"}
    )
    post_projection: bool = field(
        default=False,
        metadata={"help": "whether to apply a projection after the the Adaptor"}
    )
    projection_dim: int = field(
        default=8192,
        metadata={"help": "projection dimension"}
    )
    activation_fn: str = field(
        default="glu",
        metadata={"help": "activation function for the Adaptor"}
    )
    use_final_layer_norm: bool = field(
        default=False,
        metadata={"help": "whether to apply layer normalization to the output of the Adaptor"}
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "dropout rate for the Adaptor"}
    )

class Adaptor(nn.Module):
    def __init__(self, cfg: AdaptorConfig):
        super().__init__()
        self.pre_proj, self.post_proj = None, None
        if cfg.pre_projection:
            self.pre_proj = nn.Sequential(
                LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.projection_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout_rate),
                nn.Linear(cfg.projection_dim, cfg.embed_dim)
            )
        if cfg.post_projection:
            self.post_proj = nn.Sequential(
                LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, cfg.projection_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout_rate),
                nn.Linear(cfg.projection_dim, cfg.embed_dim),
            )
        self.dropout = FairseqDropout(cfg.dropout_rate)

        dim_factor = 2 if cfg.activation_fn == 'glu' else 1
        self.layers = nn.ModuleList(
            nn.Conv1d(
                cfg.embed_dim,
                cfg.embed_dim * dim_factor,
                cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.kernel_size // 2,
            )
            for _ in range(cfg.num_layers)
        )
        self.conv1d_layer_norm = LayerNorm(cfg.embed_dim)
        self.activation_fn = partial(nn.functional.glu, dim=1) \
            if cfg.activation_fn == 'glu' else \
            utils.get_activation_fn(cfg.activation_fn)
        self.stride = cfg.stride
        
        self.final_layer_norm = LayerNorm(cfg.embed_dim) if cfg.use_final_layer_norm else None

    def forward(self, x, padding_mask: Optional[torch.Tensor]):

        if self.pre_proj is not None:
            x = x + self.dropout(self.pre_proj(x))

        if padding_mask is not None:
            x = utils.index_put(x, padding_mask.T, 0)
            
        x = self.conv1d_layer_norm(x)

        # T x B x C -> B x C x T
        x = x.transpose(0, 1).transpose(1, 2)
        out_lens = None
        if padding_mask is not None:
            out_lens = (~padding_mask).sum(1).float()
        
        for layer in self.layers:
            x = self.activation_fn(layer(x))
            if padding_mask is not None:
                out_lens = ((out_lens - 1) / self.stride + 1).floor()
        # B x C x T -> T x B x C
        x = x.transpose(1, 2).transpose(0, 1)
        x = self.dropout(x)

        if self.post_proj is not None:
            x = x + self.dropout(self.post_proj(x))
            
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
            
        out_padding_mask = None
        if padding_mask is not None:
            out_padding_mask = lengths_to_padding_mask(out_lens.long())
            x = utils.index_put(x, out_padding_mask.T, 0)

        return x, out_padding_mask