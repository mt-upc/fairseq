from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LocalMultiheadAttention
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase


class LocalTransformerEncoderLayerBase(TransformerEncoderLayerBase):
    def __init__(self, cfg, attention_window):
        self.attention_window = attention_window
        super().__init__(cfg)

    def build_self_attention(self, embed_dim, cfg):
        return LocalMultiheadAttention(
            window_size=self.attention_window//2,
            causal=False,
            look_backward=1,
            look_forward=0,
            dropout=cfg.attention_dropout,
            embed_dim=embed_dim,
            autopad=True,
            num_heads=cfg.encoder.attention_heads,
        )


# backward compatible with the legacy argparse format
class LocalTransformerEncoderLayer(LocalTransformerEncoderLayerBase):
    def __init__(self, args, attention_window):
        super().__init__(TransformerConfig.from_namespace(args), attention_window)
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )
