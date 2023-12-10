from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
from torch import nn

from fairseq.data.data_utils import lengths_to_padding_mask, get_lengths
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import LayerNorm, PositionalEmbedding


@dataclass
class SpeechEmbedderConfig(FairseqDataclass):
    use_special_embedding: bool = field(
        default=False, metadata={"help": "use BOS and EOS embedding"}
    )
    use_positional_embedding: bool = field(
        default=False, metadata={"help": "use positional embedding"}
    )
    learned_positional_embedding: bool = field(
        default=False, metadata={"help": "use learned positional embedding"}
    )
    embed_dim: int = field(
        default=II("model.embed_dim"), metadata={"help": "embedding dimension"}
    )
    max_source_positions: int = field(
        default=II("task.max_positions_text"), metadata={"help": "maximum sequence length for learned positional embeddings"}
    )
    padding_idx: int = field(
        default=1, metadata={"help": "padding index"}
    )
    layer_norm_special: bool = field(
        default=False, metadata={"help": "use layer norm for special embedding"}
    )
    freeze: bool = field(
        default=False, metadata={"help": "freeze speech embedder"}
    )
    scale_embedding: bool = field(
        default=False, metadata={"help": "scale embeddings by sqrt(dimension)"}
    )
    learned_scale: bool = field(
        default=False, metadata={"help": "learn the scale"}
    )
    inverse_scale: bool = field(
        default=False, metadata={"help": "scale-down positional instead of scale-up embedding"}
    )
    scale_init: float = field(
        default=32.0, metadata={"help": "scale initialization"}
    )


class SpeechEmbedder(nn.Module):
    def __init__(self, cfg: SpeechEmbedderConfig):
        super().__init__()

        self.cfg = cfg
        if cfg.use_special_embedding:
            self.bos_emb = nn.Parameter(torch.zeros(cfg.embed_dim))
            self.eos_emb = nn.Parameter(torch.zeros(cfg.embed_dim))
            if cfg.layer_norm_special:
                self.layernorm_special = LayerNorm(cfg.embed_dim)
        if cfg.use_positional_embedding:
            self.pos_emb = PositionalEmbedding(
                cfg.max_source_positions,
                cfg.embed_dim,
                cfg.padding_idx,
                learned=cfg.learned_positional_embedding,
            )
            if cfg.learned_positional_embedding:
                self.embedding_layernorm = LayerNorm(cfg.embed_dim)
        if cfg.scale_embedding and cfg.learned_scale:
            self.scale = nn.Parameter(torch.tensor([cfg.scale_init]))
        elif cfg.scale_embedding:
            self.scale = cfg.scale_init
        
    def forward(self, x, padding_mask=None):
        """Add special embedding and positional embedding.
        Args:
            x (FloatTensor): (B, T, C)
            padding_mask (ByteTensor): (B, T)
        Outputs:
            x (FloatTensor): (B, T+2, C)
            padding_mask (ByteTensor): (B, T+2)
        """
        B = x.size(0)
        lengths = get_lengths(x.transpose(0, 1), padding_mask)
        assert B == len(lengths)
        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        
        if self.cfg.use_special_embedding:
            if hasattr(self, "layernorm_special"):
                bos_emb = self.layernorm_special(self.bos_emb)
                eos_emb = self.layernorm_special(self.eos_emb)
            else:
                bos_emb = self.bos_emb
                eos_emb = self.eos_emb
                
            # prepend bos
            x = torch.cat([bos_emb.view(1, 1, -1).expand(B, 1, -1), x], dim=1)
            lengths += 1
            
            # append padding (zeros) and then convert first padding to eos
            x = torch.cat([x, torch.zeros(B, 1, x.size(-1), device=x.device, dtype=x.dtype)], dim=1)
            for i in range(B):
                x[i, lengths[i], :] = eos_emb
            lengths += 1
            
            padding_mask = lengths_to_padding_mask(lengths)
        
        if self.cfg.use_positional_embedding:
            if self.cfg.inverse_scale:
                x = x + self.pos_emb(padding_mask.long()) / self.scale
            else:
                x = x * self.scale + self.pos_emb(padding_mask.long())
            if hasattr(self, "embedding_layernorm"):
                x = self.embedding_layernorm(x)
            
        return x, padding_mask, lengths