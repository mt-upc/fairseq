from dataclasses import dataclass, field

import torch
from torch import nn

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import LayerNorm, PositionalEmbedding


@dataclass
class EmbedderConfig(FairseqDataclass):
    use_special_embedding: bool = field(
        default=False, metadata={"help": "use BOS and EOS embedding"}
    )
    use_positional_embedding: bool = field(
        default=False, metadata={"help": "use positional embedding"}
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


class Embedder(nn.Module):
    def __init__(self, cfg: EmbedderConfig):
        super().__init__()

        self.cfg = cfg
        if cfg.use_special_embedding:
            self.bos_emb = nn.Parameter(torch.zeros(cfg.embed_dim))
            self.eos_emb = nn.Parameter(torch.zeros(cfg.embed_dim))
        if cfg.use_positional_embedding:
            self.pos_emb = PositionalEmbedding(
                cfg.max_source_positions,
                cfg.embed_dim,
                cfg.padding_idx,
                learned=True,
            )
            self.layernorm = LayerNorm(cfg.embed_dim)
            
    def forward(self, x, padding_mask, lengths):
        """Add special embedding and positional embedding.
        Args:
            x (FloatTensor): (B, T, C)
            padding_mask (ByteTensor): (B, T)
            lengths (LongTensor): (B)
        Outputs:
            x (FloatTensor): (B, T+2, C)
            padding_mask (ByteTensor): (B, T+2)
            lengths (LongTensor): (B)
        """
        
        B = x.size(0)
        assert B == len(lengths)
        if padding_mask is not None:
            assert B == padding_mask.size(0)
        
        if self.cfg.use_special_embedding:
            x = torch.cat([self.bos_emb.view(1, 1, -1).expand(B, 1, -1), x], dim=1)
            lengths += 1
            x = torch.cat([x, torch.zeros(B, 1, x.size(-1), device=x.device, dtype=x.dtype)], dim=1)
            for i in range(B):
                x[i, lengths[i], :] = self.eos_emb
            lengths += 1
            
            padding_mask = lengths_to_padding_mask(lengths)
        
        if self.cfg.use_positional_embedding:
            x = x + self.pos_emb(padding_mask.long())
            x = self.layernorm(x)
            
        return x, padding_mask, lengths