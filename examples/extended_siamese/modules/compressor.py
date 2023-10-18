import logging
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

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


class TransformerEncoderLayers(nn.Module):
    def __init__(self, embed_dim, num_layers, dropout_rate):
        super().__init__()

        # TODO: careful bad hardcoding
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

        x = x.transpose(0, 1)

        return x

class BasicPooling(nn.Module):
    def __init__(self, pooling_fn: str, embed_dim: int, layernorm=False):
        super().__init__()
        self.out = nn.Linear(embed_dim, embed_dim)
        self.pooling_fn = pooling_fn
        if layernorm:
            self.layer_norm = LayerNorm(embed_dim)
        
    def forward(self, x, mask):
        # x: [B, N, D]
        # mask: [B, N]
        
        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)
        
        if self.pooling_fn == "mean":
            lens = (~mask).to(torch.long).sum(dim=1).to(x.dtype) # [B]
            lens[lens.eq(0.0)] = 1.5  # to avoid division by 0
            y = torch.sum(x, dim=1) / lens.unsqueeze(1) # [B, D]
        elif self.pooling_fn == "max":
            y = torch.max(x, dim=1)[0] # [B, D]
            
        y = self.out(y) # [B, D]
            
        return y
    
class CLSPooling(nn.Module):
    def __init__(self, embed_dim, num_transformer_layers=1, dropout_rate=0.0):
        super().__init__()
        self.cls_token = torch.empty(1, 1, embed_dim)
        nn.init.xavier_uniform_(self.cls_token)
        
        self.transformer = TransformerEncoderLayers(embed_dim, num_transformer_layers, dropout_rate)
        
    def forward(self, x, mask):
        # x: [B, N, D]
        # mask: [B, N]
        
        cls_token = self.cls_token.repeat(x.size(0), 1, 1).to(dtype=x.dtype, device=x.device) # [B, 1, D]
        cls_mask = torch.zeros(x.size(0), 1, dtype=mask.dtype, device=mask.device) # [B, 1]
        x = torch.cat([cls_token, x], dim=1) # [B, N+1, D]
        mask = torch.cat([cls_mask, mask], dim=1) # [B, N+1]
        
        x = self.transformer(x, mask) # [B, N+1, D]
        
        y = x[:, 0] # [B, D]
        
        return y
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            FairseqDropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask):
        # x: [B, N, D]
        # mask: [B, N]

        attn_scores = self.attention(x).squeeze(-1) # [B, N]
        attn_scores = attn_scores.masked_fill(mask, float('-inf')) # [B, N]

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1) # [B, N, 1]

        y = torch.sum(attn_weights * x, dim=1) # [B, D]
        y = self.out(y) # [B, D]

        return y


@dataclass
class LevelCompressorConfig(FairseqDataclass):
    pooling_fn: ChoiceEnum(["mean", "max", "attention", "cls"]) = field(
        default="mean",
        metadata={"help": "pooling function for collapsing representations"},
    )
    attn_hidden_dim: int = field(
        default=4096,
        metadata={"help": "hidden dimension for attention pooling"}
    )
    cls_transformer_layers: int = field(
        default=1,
        metadata={"help": "number of transformer layers if pooling_fn is cls"},
    )
    post_pooling_adaptor: bool = field(
        default=False, metadata={"help": "whether to use an adaptor after pooling"}
    )
    adaptor_dim: int = field(
        default=4096,
        metadata={"help": "projection dimension for post-pooling adaptor"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout rate for every module in this level compressor"},
    )
    layernorm: bool = field(
        default=False,
        metadata={"help": "whether to use layer normalization before pooling"},
    )


@dataclass
class CompressorConfig(FairseqDataclass):
    embed_dim: int = field(
        default=II("model.embed_dim"), metadata={"help": "embedding dimension for char-level compressor"}
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
        self.char_pooling_module, self.char_post_adaptor = self._init_modules(cfg.char_compression)

        if cfg.token_compression is not None:
            self.token_pooling_module, self.token_post_adaptor = self._init_modules(cfg.token_compression)

    def _init_modules(self, lvl_cfg):

        if lvl_cfg.pooling_fn == "attention":
            pooling_module = AttentionPooling(self.embed_dim, lvl_cfg.attn_hidden_dim, lvl_cfg.dropout)    
        elif lvl_cfg.pooling_fn == "cls":
            pooling_module = CLSPooling(self.embed_dim, lvl_cfg.cls_transformer_layers, lvl_cfg.dropout) 
        else: # mean or max
            pooling_module = BasicPooling(lvl_cfg.pooling_fn, self.embed_dim, lvl_cfg.layernorm)

        post_adaptor = None
        if lvl_cfg.post_pooling_adaptor:
            post_adaptor = CompressionAdaptor(
                self.embed_dim, lvl_cfg.adaptor_dim, lvl_cfg.dropout
            )

        return pooling_module, post_adaptor

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
        valid_chars = compr_preds.ne(self.blank_idx)

        # create a tensor to fill-in
        n = counts.size(1) # max number of predictions for the sequences
        m = counts[valid_chars].max().item() # max length of the non-blank predictions
        x_chars = torch.zeros(B, n, m, D, device=x.device, dtype=x.dtype)

        for i in range(B):
            x_chars_i = torch.split(x[i], counts_list[i].tolist()) # Tuple[2d tensor]
            x_chars_i = pad_sequence(
                x_chars_i, batch_first=True, padding_value=0
            )  # n_i x m_i x D
            # we dont care about the length of the blank predictions
            if x_chars_i.size(1) > m:
                x_chars_i = x_chars_i[:, :m]
            x_chars[i, :x_chars_i.size(0), :x_chars_i.size(1)] = x_chars_i

        x_chars = x_chars.view(B * n, m, D)
        counts = counts.view(B * n)
        valid_chars = valid_chars.view(B * n)
        
        x_compr = torch.zeros(B * n, D, device=x.device, dtype=x.dtype)
        x_compr[valid_chars] = self.char_pooling_module(
            x_chars[valid_chars],
            lengths_to_padding_mask(counts[valid_chars])
        )  # B * n x D

        x_compr = x_compr.view(B, n, D)  # B x n x D

        compr_valid_mask = compr_preds != self.blank_idx  # B x n
        compr_valid_lens = compr_valid_mask.sum(dim=-1)  # B
        
        # correction for completelly empty sequence
        empty_examples = compr_valid_lens == 0
        if empty_examples.any():
            compr_valid_mask[empty_examples, 0] = True
            compr_valid_lens[empty_examples] = 1
            compr_preds[empty_examples, 0] = self.blank_idx

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

        if self.char_post_adaptor is not None:
            x_compr = self.char_post_adaptor(x_compr)

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
        x_tokens = torch.zeros(B, n, m, D, device=x.device, dtype=x.dtype)

        for i in range(B):
            if lens[i] > 0:
                x_tokens_i = torch.split(
                    x[i, :lens[i]], token_lengths_list[i].tolist()
                )  # Tuple[2d tensor]
            x_tokens_i = pad_sequence(
                x_tokens_i, batch_first=True, padding_value=0
            )  # n_i x m_i x D
            x_tokens[i, : x_tokens_i.size(0), : x_tokens_i.size(1)] = x_tokens_i

        x_tokens = x_tokens.view(B * n, m, D)
        token_lengths = token_lengths.view(B * n)
        token_padding_mask = lengths_to_padding_mask(token_lengths)
        valid_tokens = token_padding_mask.eq(0).any(dim=-1)

        x_compr = torch.zeros(B * n, D, device=x.device, dtype=x.dtype)
        x_compr[valid_tokens] = self.token_pooling_module(
            x_tokens[valid_tokens],
            lengths_to_padding_mask(token_lengths[valid_tokens])
        )  # B * n x D
        
        x_compr = x_compr.view(B, n, D)  # B x num_words x D
        x_compr_lens = (token_lengths.view(B, -1) != 0).sum(dim=-1)  # B
        x_compr_mask = lengths_to_padding_mask(x_compr_lens)  # B x num_words
        x_compr.masked_fill_(x_compr_mask.unsqueeze(-1), 0)

        if self.token_post_adaptor is not None:
            x_compr = self.token_post_adaptor(x_compr)
            
        return x_compr, x_compr_mask, x_compr_lens
            
    def forward(self, **kwargs):
        raise NotImplementedError("Use the compression method instead")