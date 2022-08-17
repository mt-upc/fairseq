from typing import List

from torch import nn, LongTensor

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout


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
        x = x.permute(1, 2, 0).contiguous()  # T x B x C-> B x C x T
        x = self.conv(x)
        x = x.permute(2, 0, 1).contiguous()  # B x C x T -> T x B x C
        return x


class MultiHeadPooledAttention(MultiheadAttention):
    def __init__(self, cfg):
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
    def __init__(self, cfg):
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
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.ModuleList(
            [ModalityAdapterLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.normalize_before = cfg.normalize_before
        self.layer_norm = LayerNorm(cfg.embed_dim)

    def forward(self, x, x_len):
        if not self.normalize_before:
            x = self.layer_norm(x)
        for layer in self.layers:
            x, x_len, x_mask = layer(x, x_len)
        if self.normalize_before:
            x = self.layer_norm(x)
        return x, x_mask
