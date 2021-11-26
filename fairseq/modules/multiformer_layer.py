from fairseq.models.transformer import TransformerConfig
from fairseq.modules import MultiheadMultiAttention
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase


class MultiformerEncoderLayerBase(TransformerEncoderLayerBase):
    def __init__(self, cfg, full_att_heads, fast_att_heads, local_att_cfg, compressed_att_cfg):
        self.full_att_heads=full_att_heads
        self.fast_att_heads=fast_att_heads
        self.local_att_cfg=local_att_cfg
        self.compressed_att_cfg=compressed_att_cfg
        super().__init__(cfg)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadMultiAttention(
            embed_dim=embed_dim,
            full_att_heads=self.full_att_heads,
            fast_att_heads=self.fast_att_heads,
            local_att_cfg=self.local_att_cfg,
            compressed_att_cfg=self.compressed_att_cfg,
            dropout=cfg.attention_dropout,
        )

# backward compatible with the legacy argparse format
class MultiformerEncoderLayer(MultiformerEncoderLayerBase):
    def __init__(self, args, full_att_heads, fast_att_heads, local_att_cfg, compressed_att_cfg):
        super().__init__(TransformerConfig.from_namespace(args), full_att_heads, fast_att_heads, local_att_cfg, compressed_att_cfg)
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )
