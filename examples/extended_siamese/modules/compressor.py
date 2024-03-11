import logging
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.modules import FairseqDropout, LayerNorm, PositionalEmbedding
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
        cfg.encoder.attention_heads = 16

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
    def __init__(self, pooling_fn: str, embed_dim: int, output_layer=False):
        super().__init__()
        
        self.pooling_fn = pooling_fn
        
        if self.pooling_fn == "mean-max":
            assert output_layer
            self.out = nn.Linear(embed_dim*2, embed_dim)
        else:
            if output_layer:
                self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, lens):
        # x: [B, N, D]
        # lens: [B]
        
        if self.pooling_fn == "mean":
            x = torch.sum(x, dim=1) / lens.to(dtype=x.dtype).unsqueeze(1) # [B, D]
        elif self.pooling_fn == "max":
            x = torch.max(x, dim=1)[0] # [B, D]
        elif self.pooling_fn == "mean-max":
            x = torch.cat([torch.sum(x, dim=1) / lens.to(dtype=x.dtype).unsqueeze(1), torch.max(x, dim=1)[0]], dim=-1)
        
        if hasattr(self, "out"):
            x = self.out(x) # [B, D]
            
        return x
    
class CLSPooling(nn.Module):
    def __init__(self, embed_dim, num_transformer_layers=1, dropout_rate=0.0, use_positional=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.25) # to match the norm of x
        
        self.transformer = TransformerEncoderLayers(embed_dim, num_transformer_layers, dropout_rate)
        
        if use_positional:
            self.pos_emb = PositionalEmbedding(512, embed_dim, 1)
            self.scale = math.sqrt(embed_dim)

    def forward(self, x, lens):
        # x: [B, N, D]
        # mask: [B, N]

        # prepend cls token
        x = torch.cat(
            [
                self.cls_token.to(dtype=x.dtype, device=x.device).repeat(x.size(0), 1, 1), # B x 1 x D
                x
            ],
        dim=1) # [B, N+1, D]
        
        mask = lengths_to_padding_mask(lens+1)
        
        if hasattr(self, "pos_emb"):
            x = x + self.pos_emb(mask.long()) / self.scale
        
        x = self.transformer(x, mask) # [B, N+1, D]
        x = x[:, 0] # [B, D]
        
        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            FairseqDropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, lens):
        # x: [B, N, D]
        # lens: [B]
        
        mask = lengths_to_padding_mask(lens)

        attn_scores = self.attention(x).squeeze(-1) # [B, N]
        attn_scores = attn_scores.masked_fill(mask, float('-inf')) # [B, N]

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1) # [B, N, 1]

        x = torch.sum(attn_weights * x, dim=1) # [B, D]
        x = self.out(x) # [B, D]

        return x


@dataclass
class LevelCompressorConfig(FairseqDataclass):
    pooling_fn: ChoiceEnum(["mean", "max", "mean-max", "attention", "cls"]) = field(
        default="mean",
        metadata={"help": "pooling function for collapsing representations"},
    )
    attn_hidden_dim: int = field(
        default=8192,
        metadata={"help": "hidden dimension for attention pooling"}
    )
    transformer_layers: int = field(
        default=0,
        metadata={"help": "number of transformer layers if pooling_fn is cls, otherwise separate transformer"},
    )
    post_pooling_adaptor: bool = field(
        default=False, metadata={"help": "whether to use an adaptor after pooling"}
    )
    adaptor_dim: int = field(
        default=8192,
        metadata={"help": "projection dimension for post-pooling adaptor"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout rate for every module in this level compressor"},
    )
    output_layer: bool = field(
        default=False, metadata={"help": "whether to use a linear layer after pooling"}
    )
    use_positional_embedding: bool = field(
        default=False, metadata={"help": "use positional embedding in the CLS pooling transformer"}
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
    path: Optional[str] = field(
        default=None, metadata={"help": "path to load the compressor from"}
    )
    remove_sep: bool = field(
        default=False, metadata={"help": "remove separator before pooling"}
    )


class Compressor(nn.Module):
    def __init__(self, cfg: CompressorConfig, blank_idx: int = 0, sep_idx: int = 4):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.blank_idx = blank_idx
        self.sep_idx = sep_idx
        
        self.remove_sep = False
        if hasattr(self.cfg, "remove_sep") and self.cfg.remove_sep:
            self.remove_sep = True

        assert cfg.char_compression is not None
        self.char_pooling_module, self.char_post_adaptor, self.char_pre_transformer = self._init_modules(cfg.char_compression)

        if cfg.token_compression is not None:
            self.token_pooling_module, self.token_post_adaptor, self.token_pre_transformer = self._init_modules(cfg.token_compression)
        
        if cfg.path is not None:
            self._load_from_pretrained(cfg.path)

    def _init_modules(self, lvl_cfg):

        if lvl_cfg.pooling_fn == "attention":
            pooling_module = AttentionPooling(self.embed_dim, lvl_cfg.attn_hidden_dim, lvl_cfg.dropout)    
        elif lvl_cfg.pooling_fn == "cls":
            assert lvl_cfg.transformer_layers > 0
            pooling_module = CLSPooling(self.embed_dim, lvl_cfg.transformer_layers, lvl_cfg.dropout, lvl_cfg.use_positional_embedding) 
        else: # mean or max or mean-max
            pooling_module = BasicPooling(lvl_cfg.pooling_fn, self.embed_dim, lvl_cfg.output_layer)

        post_adaptor = None
        if lvl_cfg.post_pooling_adaptor:
            post_adaptor = CompressionAdaptor(
                self.embed_dim, lvl_cfg.adaptor_dim, lvl_cfg.dropout
            )
            
        pre_transformer = None
        if lvl_cfg.transformer_layers > 0 and lvl_cfg.pooling_fn != "cls":
            pre_transformer = TransformerEncoderLayers(self.embed_dim, lvl_cfg.transformer_layers, lvl_cfg.dropout)

        return pooling_module, post_adaptor, pre_transformer
        
    def _load_module_state(self, module_name, compressor_ckpt):
        if hasattr(self, module_name) and getattr(self, module_name) is not None:
            ckpt_key = f"{module_name}_out" if "out" in module_name else module_name
            if ckpt_key in compressor_ckpt and compressor_ckpt[ckpt_key] is not None:
                getattr(self, module_name).load_state_dict(compressor_ckpt[ckpt_key])
                logger.info(f"Loaded {module_name} from checkpoint")
            else:
                logger.warning(f"Could not load {module_name} from checkpoint")

    def _load_from_pretrained(self, path):
        logger.info(f"Loading compressor from {path}")
        compressor_ckpt = torch.load(path)
        # List of module names to load
        module_names = [
            "char_pooling_module.out",
            "token_pooling_module.out",
            "char_post_adaptor",
            "token_post_adaptor",
            "char_pre_transformer",
            "token_pre_transformer",
            "token_pooling_module",
            "char_pooling_module"
        ]
        # Load each module
        for module_name in module_names:
            self._load_module_state(module_name, compressor_ckpt)
            

    def char_compression(self, x, preds, lens):
        # x: B x T x D
        # preds: B x T
        # lens: B
        
        B, T, D = x.size()
        device = x.device
        dtype = x.dtype
        
        # zero-out the padding
        mask = lengths_to_padding_mask(lens) # B x T
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        preds = preds.masked_fill(mask, self.blank_idx)

        # add a vector of -1 to know where each example ends after flattening the batch
        preds = torch.cat([-torch.ones(B, 1, device=device, dtype=torch.long), preds], dim=1).view(-1)
        x = torch.cat([torch.zeros(B, 1, D, device=device, dtype=dtype), x], dim=1).view(-1, D)

        # get points of consecutive preds
        preds, counts = preds.unique_consecutive(return_counts=True)
        
        # split in representations of same chars
        x = torch.split(x, counts.tolist())
        
        # remove blanks
        valid_mask = preds != self.blank_idx
        preds = preds[valid_mask]
        counts = counts[valid_mask]
        x = [x_i for x_i, v_i in zip(x, valid_mask) if v_i]
        
        # pack into tensor
        x = pad_sequence(x, batch_first=True, padding_value=0)

        # reduce dim 1
        if self.char_pre_transformer is not None:
            mask = lengths_to_padding_mask(counts)
            x = self.char_pre_transformer(x, mask)
        
        x = self.char_pooling_module(x, counts)

        # find split points for retrieving the examples
        split_points = (preds == -1).nonzero(as_tuple=True)[0]
        split_points = torch.cat([split_points, torch.tensor([len(preds)], device=device)])
        split_points = (split_points[1:] - split_points[:-1]).tolist()

        # split into examples
        x = torch.split(x, split_points)
        preds = torch.split(preds, split_points)
        lens = torch.tensor([len(x_i) for x_i in x], device=device)

        # pack into tensors
        x = pad_sequence(x, batch_first=True, padding_value=0)
        preds = pad_sequence(preds, batch_first=True, padding_value=self.blank_idx)

        # remove the parts we add to identify the bounds for each example
        x = x[:, 1:]
        preds = preds[:, 1:]
        lens -= 1

        mask = lengths_to_padding_mask(lens)
        
        # account for empty examples (just a sep token)
        empty_examples = lens == 0
        num_empty_examples = empty_examples.sum()
        if num_empty_examples > 0:
            logger.warning(f"Found {num_empty_examples} empty examples")
            mask[empty_examples, 0] = True
            lens[empty_examples] = 1
            preds[empty_examples, 0] = self.sep_idx

        # process representation
        if self.char_post_adaptor is not None:
            x = self.char_post_adaptor(x)
        
        return x, mask, lens, preds, num_empty_examples
    
    def token_compression(self, x, preds, lens):
        # x: B x T x D
        # preds: B x T
        # lens: B
        
        B, T, D = x.size()
        device = x.device
        dtype = x.dtype
        
        # new lengths after compression
        new_lens = preds.eq(self.sep_idx).sum(dim=1)
        
        # unpad and unpack to list of tensors
        preds = [preds[i, :lens[i]] for i in range(B)]
        x = [x[i, :lens[i]] for i in range(B)]
        
        # make sure every example ends with a separator
        num_examples_without_ending_sep = torch.tensor(0, device=device, dtype=torch.long)
        for i in range(B):
            if preds[i][-1] != self.sep_idx:
                logger.warning(f"Example with length {lens[i]} and new length {new_lens[i]} ending with {preds[i][-1]}")
                preds[i] = torch.cat([preds[i], torch.tensor([self.sep_idx], device=device, dtype=torch.long)])
                x[i] = torch.cat([x[i], torch.zeros(1, D, device=device, dtype=dtype)])
                new_lens[i] += 1
                num_examples_without_ending_sep += 1
        
        # flatten
        preds = torch.cat(preds)
        x = torch.cat(x)
        
        # split points according to separators
        split_points = preds.eq(self.sep_idx).nonzero(as_tuple=True)[0] + 1
        split_points = torch.cat([torch.tensor([0], device=device, dtype=torch.long), split_points])
        split_points = (split_points[1:] - split_points[:-1]).tolist()
        
        # re-arrange in 3d [total_num_tokens x max(count) x D]
        x = torch.split(x, split_points) # Tuple[2d tensor]
        if self.remove_sep:
            x = [x_i[:-1] for x_i in x] # remove sep
        counts = torch.tensor([len(x_i) for x_i in x], device=device, dtype=torch.long)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        
        if self.token_pre_transformer is not None:
            mask = lengths_to_padding_mask(counts)
            x = self.token_pre_transformer(x, mask)
        
        # reduce dim 1
        x = self.token_pooling_module(x, counts)
        
        # reconstruct the batch
        split_points = new_lens.cumsum(dim=0)
        split_points = torch.cat([torch.tensor([0], device=device, dtype=torch.long), split_points])
        split_points = (split_points[1:] - split_points[:-1]).tolist()
        x = torch.split(x, split_points)
        x = pad_sequence(x, batch_first=True, padding_value=0) # B x ? x D
        
        mask = lengths_to_padding_mask(new_lens)
        
        if self.token_post_adaptor is not None:
            x = self.token_post_adaptor(x)
        
        return x, mask, new_lens, num_examples_without_ending_sep  

    def forward(self, **kwargs):
        raise NotImplementedError("Use the compression method instead")