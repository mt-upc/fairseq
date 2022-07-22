from typing import List

from torch import nn

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.speech_to_text import Conv1dSubsampler
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout


class Conv1dLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int]
    ):
        super(Conv1dLayers, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def forward(self, x):
        x = x.permute(1, 2, 0).contiguous()  # -> B x D x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        x = x.permute(2, 0, 1).contiguous()  # -> T x B x D
        return x


class MultiHeadPooledAttention(MultiheadAttention):
    def __init__(self, cfg):
        super().__init__(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
        )
        
        self.k_proj = self._modify(self.k_proj, cfg)
        self.v_proj = self._modify(self.v_proj, cfg)
        self.q_proj = self._modify(self.q_proj, cfg)

        # to bypass the pytorch MHA
        self._set_skip_embed_dim_check()
        
    def _modify(self, proj, cfg):
        return nn.Sequential(
            proj,
            Conv1dLayers(
                in_channels=cfg.in_channels,
                mid_channels=cfg.mid_channels,
                out_channels=cfg.out_channels,
                kernel_sizes=cfg.kernel_sizes,
            ),
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

        self.input_pool = Conv1dSubsampler(
            in_channels=cfg.in_channels,
            mid_channels=cfg.mid_channels,
            out_channels=cfg.out_channels,
            kernel_sizes=cfg.kernel_sizes,
        )

        self.mhpa = MultiHeadPooledAttention(cfg)

        self.attn_layer_norm = LayerNorm(cfg.embed_dim)

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.ffn_embed_dim, cfg.embed_dim)

        self.final_layer_norm = LayerNorm(cfg.embed_dim)

    def forward(self, x, x_len):

        if self.normalize_before:
            x = self.attn_layer_norm(x)

        residual = x

        residual, x_len = self.input_pool(residual.transpose(0, 1), x_len)
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

    def forward(self, x, x_len):
        for layer in self.layers:
            x, x_len, x_mask = layer(x, x_len)
        return x, x_mask
