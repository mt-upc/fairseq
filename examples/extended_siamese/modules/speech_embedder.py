from dataclasses import dataclass, field

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
    scale_embedding: bool = field(
        default=False, metadata={"help": "scale embeddings by sqrt(dimension)"}
    )
    embed_dim: int = field(
        default=1024, metadata={"help": "embedding dimension"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "maximum sequence length for learned positional embeddings"}
    )
    padding_idx: int = field(
        default=1, metadata={"help": "padding index"}
    )
    layer_norm_special: bool = field(
        default=False, metadata={"help": "use layer norm for special embedding"}
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
                self.layernorm = LayerNorm(cfg.embed_dim)
        self.pre_scale = 7.0 if cfg.scale_embedding else 1.0
        
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
        
        if self.cfg.use_special_embedding:
            # prepend bos
            if hasattr(self, "layernorm_special"):
                bos_emb = self.layernorm_special(self.bos_emb)
                eos_emb = self.layernorm_special(self.eos_emb)
            else:
                bos_emb = self.bos_emb
                eos_emb = self.eos_emb
            x = torch.cat([bos_emb.view(1, 1, -1).expand(B, 1, -1), x], dim=1)
            lengths += 1
            # append padding (zeros) and then convert first padding to eos
            x = torch.cat([x, torch.zeros(B, 1, x.size(-1), device=x.device, dtype=x.dtype)], dim=1)
            for i in range(B):
                x[i, lengths[i], :] = eos_emb
            lengths += 1
            
            padding_mask = lengths_to_padding_mask(lengths)
        
        x *= self.pre_scale
        
        if self.cfg.use_positional_embedding:
            x = x + self.pos_emb(padding_mask.long())
            if hasattr(self, "layernorm"):
                x = self.layernorm(x)
            
        return x, padding_mask, lengths