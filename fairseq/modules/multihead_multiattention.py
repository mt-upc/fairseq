import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Tuple, List

from fairseq.modules import FullAttention, ConvAttention
from fairseq.data.data_utils import lengths_to_padding_mask

try:
    from local_attention import LocalAttention
except ImportError:
    print("Please install the local-attention package")


class MultiheadMultiAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        full_att_heads: int = 0,
        local_att_cfg: Tuple[Tuple[int]] = ((0,1),),
        compressed_att_cfg: Tuple[Tuple[int]] = ((0,1,1),),
        compressed_conv_type: str = 'normal',
        dropout: float = 0.,
    ):
        """ MultiheadMultiAttention

        Attributes:
            embed_dim: Embedding dimension
            full_att_heads: Number of full attention heads 
            local_att_cfg: Local Attention config ((n_heads, window_size), ...)
            compressed_att_cfg: Compressed Attention config ((n_heads, kernel_size, comp_factor), ...)
            compressed_conv_type: Convolutions type ('normal', 'separable' or 'depthwise')
            dropout: Dropout probability

        """
        super(MultiheadMultiAttention, self).__init__()

        self.full_att_heads = full_att_heads
        self.local_att_cfg = local_att_cfg
        self.compressed_att_cfg = compressed_att_cfg

        assert self.num_heads > 0, \
            "You must set at least one head"

        assert embed_dim % self.num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // self.num_heads

        self.attentions = nn.ModuleList([])

        if full_att_heads > 0:
            self.attentions.append(FullAttention(dropout))

        for (_, w_s) in self.local_att_cfg:
            self.attentions.append(
                LocalAttention(
                    dim=self.head_dim,
                    window_size=w_s//2,
                    causal=False,
                    look_backward=1,
                    look_forward=0,
                    dropout=dropout,
                    autopad=True,
                )
            )
            self.attentions[-1].rel_pos = None

        for (_, k_s, comp_f) in self.compressed_att_cfg:
            self.attentions.append(
                ConvAttention(
                    dim=self.head_dim,
                    kernel_size=k_s,
                    stride=comp_f,
                    conv_type=compressed_conv_type,
                    dropout=dropout,
                )
            )

        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=True)

    @property
    def num_heads_list(self):
        return [
            self.full_att_heads,
            [cfg[0] for cfg in self.local_att_cfg],
            [cfg[0] for cfg in self.compressed_att_cfg],
        ]

    @property
    def num_heads_flat_list(self):
        num_heads_flat_list = [self.num_heads_list[0],
                               *self.num_heads_list[1],
                               *self.num_heads_list[2]]
        num_heads_flat_list = [i for i in num_heads_flat_list if i != 0]
        return num_heads_flat_list

    @property    
    def num_heads(self):
        return self.num_heads_list[0] + \
               sum(self.num_heads_list[1]) + \
               sum(self.num_heads_list[2])

    def split_heads(self, x: torch.FloatTensor):
        # B x T x C -> H * (B x T x C_h)
        x = iter(x.chunk(self.num_heads, dim=-1))
        # H * (B x T x C_h) -> [((B * H_1) x T x C_h), ..., (B * H_n) x T x C_h)]
        x = [torch.cat([next(x) for _ in range(h)], dim=0)
            for h in self.num_heads_flat_list]
        return x

    def concat_heads(self, x: List[torch.FloatTensor]):
        # [((B * H_1) x T x C_h), ..., (B * H_n) x T x C_h)] -> H * (B x T x C_h)
        x = torch.cat(x, dim=0).chunk(self.num_heads, dim=0)
        # H * (B x T x C_h) -> B x T x C
        x = torch.cat(x, dim=-1)
        return x

    def forward(self, q, k, v, mask_q=None, mask_kv=None):
        # T x B x C -> B X T X C
        q, k, v = map(lambda x: x.permute(1, 0, 2), (q, k, v))
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # B x T x C -> [((B * H_1) x T x C_h), ..., (B * H_n) x T x C_h)]
        q, k, v = map(self.split_heads, (q, k, v))

        out = []
        for n_h, att, q_, k_, v_ in zip(self.num_heads_flat_list, self.attentions, q, k, v):
            mask_q_ = mask_q.tile(n_h, 1) if mask_q is not None else None
            mask_kv_ = mask_kv.tile(n_h, 1) if mask_kv is not None else None
            if isinstance(att, LocalAttention):
                assert mask_kv is not None, \
                    "Local Attention can only be used for self-attention"
                out.append(att(q_, k_, v_, ~mask_q_))
            else:
                out.append(att(q_, k_, v_, mask_q_, mask_kv_))

        # [((B * H_1) x T x C_h), ..., (B * H_n) x T x C_h)] ->  B x T x C
        out = self.concat_heads(out)
        out = self.wo(out)

        # B x T x C -> T x B x C
        out = out.permute(1, 0, 2)

        return out
