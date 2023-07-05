import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.modules import FairseqDropout, LayerNorm
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase

logger = logging.getLogger(__name__)


class CompressionAdaptor(nn.Module):
    def __init__(self, embed_dim, proj_dim, dropout_rate):
        super().__init__()

        self.layers = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            FairseqDropout(dropout_rate),
            nn.Linear(proj_dim, embed_dim),
        )
        self.dropout = FairseqDropout(dropout_rate)

    def forward(self, x):
        return x + self.dropout(self.layers(x))


class CompressionTransformer(nn.Module):
    def __ini__(self, embed_dim, num_layers, dropout_rate):
        super().__init__()

        cfg = TransformerConfig()
        cfg.encoder.embed_dim = embed_dim
        cfg.encoder.ffn_embed_dim = embed_dim * 4
        cfg.encoder.normalize_before = True
        cfg.dropout = dropout_rate
        cfg.attention_dropout = dropout_rate
        cfg.activation_dropout = dropout_rate
        cfg.encoder.attention_heads = 16 if embed_dim == 1024 else 8

        self.layers = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(num_layers)]
        )

    def forward(self, x, padding_mask):
        assert x.size(0) == padding_mask.size(0)
        x = x.transpose(0, 1)

        if padding_mask is not None:
            x = x * (1 - padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        for layer in self.layers:
            x = layer(x, padding_mask)

        x_words = x_words.transpose(0, 1)

        return x


class LearnedScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(LearnedScaledDotProductAttention, self).__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim**0.5

    def forward(self, x, padding_mask=None):
        # x: [B, N, D], mask: [B, N]

        # Compute query, key and value: apply learned linear transformations
        q = self.query(x)  # [B, N, D]
        k = self.key(x)  # [B, N, D]
        v = self.value(x)  # [B, N, D]

        # Compute dot product between query q and key k, and scale it
        scores = torch.einsum("bnd,bnd->bn", q, k) / self.scale  # [B, N]

        # Apply mask - set the scores to a large negative value where mask is True
        if padding_mask is not None:
            scores.masked_fill_(padding_mask, float("-inf"))

        # Apply softmax to get the attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, N]

        # Compute the weighted sum of values along the sequence dimension
        y = torch.einsum("bn,bnd->bd", attn_weights, v)  # [B, D]

        y = self.out(y)

        return y


@dataclass
class LevelCompressorConfig(FairseqDataclass):
    pooling_fn: ChoiceEnum(["mean", "max", "attention"]) = field(
        default="mean",
        metadata={"help": "pooling function for collapsing representations"},
    )
    pre_pooling_processor: ChoiceEnum(["none", "adaptor", "transformer"]) = field(
        default="none",
        metadata={"help": "pooling processor for this level compression"},
    )
    post_pooling_adaptor: bool = field(
        default=False, metadata={"help": "whether to use an adaptor after pooling"}
    )
    transformer_layers: int = field(
        default=1,
        metadata={
            "help": "number of transformer layers if pooling_processor is transformer"
        },
    )
    adaptor_dim: int = field(
        default=8192,
        metadata={"help": "projection dimension if pooling_processor is adaptor"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout rate for every module in this level compressor"},
    )


@dataclass
class CompressorConfig(FairseqDataclass):
    embed_dim: int = field(
        default=1024, metadata={"help": "embedding dimension for char-level compressor"}
    )
    char_compression: Optional[LevelCompressorConfig] = field(
        default=None, metadata={"help": "configuration for char-level compression"}
    )
    token_compression: Optional[LevelCompressorConfig] = field(
        default=None, metadata={"help": "configuration for token-level compression"}
    )


class Compressor(nn.Module):
    def __init__(self, cfg: CompressorConfig, blank_idx: int = 0, sep_idx: int = 4):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.blank_idx = blank_idx
        self.sep_idx = sep_idx

        assert cfg.char_compression is not None
        (
            self.char_attention_pooling,
            self.char_pre_processor,
            self.char_post_processor,
        ) = self._init_modules(cfg.char_compression)

        if cfg.token_compression is not None:
            (
                self.token_attention_pooling,
                self.token_pre_processor,
                self.token_post_processor,
            ) = self._init_modules(cfg.token_compression)

    def _init_modules(self, lvl_cfg):
        attention_pooling = None
        if lvl_cfg.pooling_fn == "attention":
            attention_pooling = LearnedScaledDotProductAttention(self.embed_dim)

        pre_processor = None
        if lvl_cfg.pre_pooling_processor == "transformer":
            pre_processor = CompressionTransformer(
                self.embed_dim, lvl_cfg.transformer_layers, lvl_cfg.dropout
            )
        elif lvl_cfg.pre_pooling_processor == "adaptor":
            pre_processor = CompressionAdaptor(
                self.embed_dim, lvl_cfg.adaptor_dim, lvl_cfg.dropout
            )

        post_processor = None
        if lvl_cfg.post_pooling_adaptor:
            post_processor = CompressionAdaptor(
                self.embed_dim, lvl_cfg.adaptor_dim, lvl_cfg.dropout
            )

        return attention_pooling, pre_processor, post_processor

    def pool(self, x, lens=None, lvl="char"):
        if lvl == "char":
            fn = self.cfg.char_compression.pooling_fn
        else:
            fn = self.cfg.token_compression.pooling_fn

        if fn == "mean":
            lens_ = lens.to(x.dtype)
            lens_[lens_ == 0.0] = 1.5  # to avoid division by 0
            y = torch.sum(x, dim=1) / lens_.unsqueeze(1)

        elif fn == "max":
            y = torch.max(x, dim=1)[0]

        elif fn == "attention":
            if lvl == "char":
                attention_pooling = self.char_attention_pooling
            else:
                attention_pooling = self.token_attention_pooling

            mask = lengths_to_padding_mask(lens)
            y = torch.zeros(x.size(0), x.size(-1), dtype=x.dtype, device=x.device)
            valid_mask = mask.eq(0).any(dim=1)
            y[valid_mask] = attention_pooling(x[valid_mask], mask[valid_mask])

        return y

    def pre_process(self, x, padding_mask, valid_examples, processor):
        if processor is None:
            return x

        if isinstance(processor, CompressionTransformer):
            x[valid_examples] = processor(
                x[valid_examples], padding_mask[valid_examples]
            )
        elif isinstance(processor, CompressionAdaptor):
            x[valid_examples] = processor(x[valid_examples])

        return x
    
    def char_compression(self, x, preds):
        # x: B x T x D
        # preds: B x T

        B, T, D = x.size()
        
        # get the unique consecutive elements and their counts
        counts_list = []
        compr_preds = []
        for i in range(B):
            # p: compressed sequence, [N_i]
            # c: repeating counts for each j element of p, [N_i]
            p, c = preds[i].unique_consecutive(return_counts=True)
            counts_list.append(c)
            compr_preds.append(p)

        counts = pad_sequence(counts_list, batch_first=True, padding_value=0)
        compr_preds = pad_sequence(
            compr_preds, batch_first=True, padding_value=self.blank_idx
        )

        # create a tensor to fill-in
        n = counts.size(1)
        m = counts.max()
        x_compr = torch.zeros(B, n, m, D, device=x.device, dtype=x.dtype)

        for i in range(B):
            if i == 47:
                breakpoint()
            x_chars_i = torch.split(x[i], counts_list[i].tolist()) # Tuple[2d tensor]
            x_chars_i = pad_sequence(
                x_chars_i, batch_first=True, padding_value=0
            )  # n_i x m_i x D
            x_compr[i, : x_chars_i.size(0), : x_chars_i.size(1)] = x_chars_i

        x_compr = x_compr.view(B * n, m, D)
        counts = counts.view(B * n)
        char_padding_mask = lengths_to_padding_mask(counts)
        valid_examples = char_padding_mask.eq(0).any(dim=-1)

        x_compr = self.pre_process(
            x_compr, char_padding_mask, valid_examples, self.char_pre_processor
        )

        x_compr = self.pool(x_compr, lens=counts, lvl="char")  # B * n x D

        x_compr = x_compr.view(B, n, D)  # B x n x D

        compr_valid_mask = compr_preds != self.blank_idx  # B x n
        compr_valid_lens = compr_valid_mask.sum(dim=-1)  # B

        # remove blanks
        x_compr = torch.split(
            x_compr[compr_valid_mask], compr_valid_lens.tolist()
        )  # Tuple[tensor] of size B
        x_compr = pad_sequence(x_compr, batch_first=True, padding_value=0)  # B x N x D
        compr_preds = torch.split(
            compr_preds[compr_valid_mask], compr_valid_lens.tolist()
        )  # Tuple[tensor] of size B
        compr_preds = pad_sequence(
            compr_preds, batch_first=True, padding_value=self.blank_idx
        )  # B x N

        mask = lengths_to_padding_mask(compr_valid_lens)  # B x N

        if self.char_post_processor is not None:
            x_compr = self.char_post_processor(x_compr)

        return x_compr, mask, compr_valid_lens, compr_preds
    
    def token_compression(self, x, lens, preds):
        # x: B x T x D
        # lens: B
        # preds: B x T

        sep_mask = preds.eq(self.sep_idx)  # B x T

        B, T, D = x.size()
        dev = x.device

        # get token lengths
        token_lengths_list = []
        for i in range(B):
            # Find the indices where the separators are present in the sentence
            sep_indices = sep_mask[i].nonzero().squeeze(1)

            # account for missing last separator
            no_last_sep = preds[i, lens[i] - 1] != self.sep_idx
            if no_last_sep:
                sep_indices = torch.cat(
                    [sep_indices, torch.tensor([lens[i]], device=dev)]
                )

            # account for consecutive separators
            diffs = torch.diff(sep_indices, append=sep_indices[-1].unsqueeze(0) + 2)
            transitions = diffs != 1
            if transitions.any():
                # Select only the first element of each set of consecutive numbers
                sep_indices = sep_indices[transitions]

            # Compute token lengths
            sep_indices[: lens[i]] += 1
            zero_tensor = torch.tensor([0], device=dev)
            token_lengths = torch.diff(sep_indices, prepend=zero_tensor)
            if no_last_sep:
                token_lengths[-1] -= 1

            token_lengths_list.append(token_lengths)

        token_lengths = pad_sequence(
            token_lengths_list, batch_first=True, padding_value=0
        )

        # create a tensor to fill-in
        n = token_lengths.size(1)
        m = token_lengths.max()
        x_compr = torch.zeros(B, n, m, D, device=x.device, dtype=x.dtype)

        for i in range(B):
            if lens[i] > 0:
                x_tokens_i = torch.split(
                    x[i, :lens[i]], token_lengths_list[i].tolist()
                )  # Tuple[2d tensor]
            x_tokens_i = pad_sequence(
                x_tokens_i, batch_first=True, padding_value=0
            )  # n_i x m_i x D
            x_compr[i, : x_tokens_i.size(0), : x_tokens_i.size(1)] = x_tokens_i

        x_compr = x_compr.view(B * n, m, D)
        token_lengths = token_lengths.view(B * n)
        token_padding_mask = lengths_to_padding_mask(token_lengths)
        valid_examples = token_padding_mask.eq(0).any(dim=-1)

        x_compr = self.pre_process(
            x_compr, token_padding_mask, valid_examples, self.token_pre_processor
        )

        x_compr = self.pool(x_compr, lens=token_lengths, lvl="token")  # B * n x D

        x_compr = x_compr.view(B, n, D)  # B x num_words x D
        x_compr_lens = (token_lengths.view(B, -1) != 0).sum(dim=-1)  # B
        x_compr_mask = lengths_to_padding_mask(x_compr_lens)  # B x num_words
        x_compr.masked_fill_(x_compr_mask.unsqueeze(-1), 0)

        if self.token_post_processor is not None:
            x_compr = self.token_post_processor(x_compr)
            
        return x_compr, x_compr_mask, x_compr_lens
            
    def forward(self, x):
        raise NotImplementedError("Use the compression method instead")