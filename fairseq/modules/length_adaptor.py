from functools import partial
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

import torch
from torch import nn, LongTensor

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout


@dataclass
class Conv1dAdaptorConfig(FairseqDataclass):
    in_channels: int = field(
        default=1024,
        metadata={"help": "# of input channels in the Conv1d Adaptor"}
    )
    out_channels: int = field(
        default=1024,
        metadata={"help": "# of output channels in the Conv1d Adaptor"}
    )
    mid_channels: int = field(
        default=1024,
        metadata={"help": "# of intermediate channels in the Conv1d Adaptor"}
    )
    num_layers: int = field(
        default=1,
        metadata={"help": "# of Conv1d layers"}
    )
    kernel_size: int = field(
        default=3,
        metadata={"help": "kernel size of each Conv1d in the Conv1d Adaptor"}
    )
    stride: int = field(
        default=2,
        metadata={"help": "stride of each Conv1d in the Conv1d Adaptor"}
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "whether to use LayerDrop in the Conv1d Adaptor"}
    )
    layernorm: bool = field(
        default=False,
        metadata={"help": "whether to use LayerNorm in the Conv1d Adaptor"}
    )
    projection: bool = field(
        default=False,
        metadata={"help": "whether to apply projections in the Conv1d Adaptor"}
    )
    activation_fn: str = field(
        default="glu",
        metadata={"help": "activation function"}
    )
    path: str = field(
        default="", metadata={"help": "path to the pretrained adaptor model"}
    )
    final_layer_norm: bool = field(
        default=False, metadata={"help": "whether to apply final layer norm"}
    )


class Conv1dAdaptor(nn.Module):
    def __init__(self, cfg: Conv1dAdaptorConfig):
        super().__init__()
        self.proj, self.proj_ln = None, None
        self.post_proj, self.post_proj_ln = None, None
        if cfg.projection:
            self.proj = nn.Sequential(
                nn.Linear(cfg.in_channels, cfg.in_channels * 4), nn.ReLU(), nn.Linear(cfg.in_channels * 4, cfg.in_channels)
            )
            self.proj_ln = LayerNorm(cfg.in_channels)
            self.post_proj = nn.Sequential(
                nn.Linear(cfg.out_channels, cfg.out_channels * 4),
                nn.ReLU(),
                nn.Linear(cfg.out_channels * 4, cfg.out_channels),
            )
            self.post_proj_ln = LayerNorm(cfg.out_channels)

        dim_factor = 2 if cfg.activation_fn == 'glu' else 1
        self.layers = nn.ModuleList(
            nn.Conv1d(
                cfg.in_channels if i == 0 else cfg.mid_channels,
                (cfg.out_channels if i == (cfg.num_layers - 1) else cfg.mid_channels) * dim_factor,
                cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.kernel_size // 2,
            )
            for i in range(cfg.num_layers)
        )
        self.activation_fn = partial(nn.functional.glu, dim=1) \
            if cfg.activation_fn == 'glu' else \
            utils.get_activation_fn(cfg.activation_fn)
        self.stride = cfg.stride
        self.layerdrop = cfg.layerdrop
        self.layernorm = LayerNorm(cfg.in_channels) if cfg.layernorm else None
        self.final_layer_norm = LayerNorm(cfg.out_channels) if cfg.final_layer_norm else None

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        if self.layernorm is not None:
            x = self.layernorm(x)

        if self.proj is not None:
            x = x + 0.5 * self.proj(x)
            x = self.proj_ln(x)

        if padding_mask is not None:
            x = utils.index_put(x, padding_mask.T, 0)

        # T x B x C -> B x C x T
        x = x.transpose(0, 1).transpose(1, 2)
        out_lens = None
        if padding_mask is not None:
            out_lens = (~padding_mask).sum(1).float()

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                x = self.activation_fn(layer(x))
                if padding_mask is not None:
                    out_lens = ((out_lens - 1) / self.stride + 1).floor()
        # B x C x T -> T x B x C
        x = x.transpose(1, 2).transpose(0, 1)

        if self.post_proj is not None:
            x = x + 0.5 * self.post_proj(x)
            x = self.post_proj_ln(x)
            
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
            
        out_padding_mask = None
        if padding_mask is not None:
            out_padding_mask = lengths_to_padding_mask(out_lens.long())
            x = utils.index_put(x, out_padding_mask.T, 0)
        return x, out_padding_mask


@dataclass
class ModalityAdapterConfig(FairseqDataclass):
    num_layers: int = field(
        default=1,
        metadata={"help": "# of Modality Adapter layers"}
    )
    kernel_size: int = field(
        default=8,
        metadata={"help": "kernel size of each Conv1d in the Pooled MHA"
                        "and input pooling of each Modality Adapter layer."}
    )
    stride: int = field(
        default=8,
        metadata={"help": "stride of each Conv1d in the Pooled MHA"
                        "and input pooling of each Modality Adapter layer."}
    )
    # NOTE: couldnt find a proper way to inherit them with II from the encoder
    embed_dim: int = field(
        default=1024,
        metadata={"help": "model dimensionality"}
    )
    ffn_embed_dim: int = field(
        default=4096,
        metadata={"help": "feed-forward layer dimensionality"}
    )
    attention_heads: int = field(
        default=16,
        metadata={"help": "activation function"}
    )
    activation_fn: str = field(
        default="gelu",
        metadata={"help": "activation function"}
    )
    # NOTE: no information on dropout rates on the paper
    # I guess we can use the same as in the rest of the model
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout rate"}
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout rate after the activation function"}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout rate after in the Pooled MHA"}
    )
    # NOTE: in figure(1) it seems they use post-LN
    # but I think it should be pre-LN because both wav2vec and mbart are pre-LN
    normalize_before: bool = field(
        default=True,
        metadata={"help": "whether to use a pre-LN architecture"}
    )


class PoolingLayer(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 8, stride: int = 8):
        super(PoolingLayer, self).__init__()
        self.d = embed_dim
        self.ksz = kernel_size
        self.st = stride
        self.pad = stride // 2
        self.conv = nn.Conv1d(self.d, self.d, self.ksz, self.st, self.pad)

    def get_out_seq_lens_tensor(self, in_lengths: LongTensor):
        in_lengths = in_lengths.float()
        out_lengths = (in_lengths + 2 * self.pad - self.ksz) / self.st + 1
        out_lengths = out_lengths.floor().long()
        return out_lengths

    def forward(self, x):
        x = x.permute(1, 2, 0).contiguous()  # T x B x C -> B x C x T
        x = self.conv(x)
        x = x.permute(2, 0, 1).contiguous()  # B x C x T -> T x B x C
        return x


class MultiHeadPooledAttention(MultiheadAttention):
    def __init__(self, cfg: ModalityAdapterConfig):
        super().__init__(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
        )

        self.embed_dim = cfg.embed_dim
        self.kernel_size = cfg.kernel_size
        self.stride = cfg.stride

        self.k_proj = self._modify(self.k_proj)
        self.v_proj = self._modify(self.v_proj)
        self.q_proj = self._modify(self.q_proj)

        # to bypass the pytorch MHA
        self._set_skip_embed_dim_check()

    def _modify(self, proj):
        return nn.Sequential(
            proj,
            PoolingLayer(self.embed_dim, self.kernel_size, self.stride),
        )

    def forward(self, x, x_mask):
        result, _ = super().forward(query=x, key=x, value=x, key_padding_mask=x_mask)
        return result


class ModalityAdapterLayer(nn.Module):
    def __init__(self, cfg: ModalityAdapterConfig):
        super().__init__()

        self.normalize_before = cfg.normalize_before

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            cfg.activation_dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)

        self.input_pool = PoolingLayer(
            cfg.embed_dim,
            cfg.kernel_size,
            cfg.stride,
        )

        self.mhpa = MultiHeadPooledAttention(cfg)

        self.attn_layer_norm = LayerNorm(cfg.embed_dim)

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.ffn_embed_dim, cfg.embed_dim)

        self.final_layer_norm = LayerNorm(cfg.embed_dim)

    def forward(self, x, x_len):

        residual = x
        if self.normalize_before:
            x = self.attn_layer_norm(x)

        # pool input and get new mask
        residual = self.input_pool(residual)
        x_len = self.input_pool.get_out_seq_lens_tensor(x_len)
        x_mask = lengths_to_padding_mask(x_len)

        x = self.mhpa(x, x_mask)
        x = self.dropout_module(x)
        x = x + residual

        if not self.normalize_before:
            x = self.attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        x = self.dropout_module(x)
        x = x + residual
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, x_len, x_mask


class ModalityAdapter(nn.Module):
    def __init__(self, cfg: ModalityAdapterConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [ModalityAdapterLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.normalize_before = cfg.normalize_before
        self.layer_norm = LayerNorm(cfg.embed_dim)

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        x_len = (~padding_mask).sum(dim=1) if padding_mask is not None \
                else torch.LongTensor([x.size(0)] * x.size(1))
        x_len = x_len.to(x.device)
        if not self.normalize_before:
            x = self.layer_norm(x)
        for layer in self.layers:
            x, x_len, x_mask = layer(x, x_len)
        if self.normalize_before:
            x = self.layer_norm(x)
        return x, x_mask
