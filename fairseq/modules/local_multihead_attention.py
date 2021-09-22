import torch
import torch.nn as nn

try:
    from local_attention import LocalAttention
except ImportError:
    print("Please install the local-attention package")

class LocalMultiheadAttention(nn.Module):
    """Local multi-headed attention.

    Based on the implementation by @lucidrains:
    https://github.com/lucidrains/local-attention
    """
    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        dropout=0.,
        shared_qk=False,
        rel_pos_emb_config=None,
        embed_dim=None,
        autopad=False,
        exact_windowsize=False,
        num_heads=None,
    ):
        super(LocalMultiheadAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention = LocalAttention(
            window_size,
            causal,
            look_backward,
            look_forward,
            dropout,
            shared_qk,
            rel_pos_emb_config,
            self.head_dim,
            autopad,
            exact_windowsize,
        )
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=True)

        self.reset_parameters()
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

        nn.init.constant_(self.wo.bias, 0.0)

    def forward(self, query, key, value, key_padding_mask, **kwargs):

        # padding_positions=True -> padding_positions=False
        input_mask = ~key_padding_mask

        # T x B x C -> B X T X C
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        query, key, value = self.wq(query), self.wk(key), self.wv(value)

        # B x T x C -> (B x num_heads) x T x head_dim)
        query = torch.cat(query.chunk(self.num_heads, dim=-1), dim=0)
        key = torch.cat(key.chunk(self.num_heads, dim=-1), dim=0)
        value = torch.cat(value.chunk(self.num_heads, dim=-1), dim=0)

        out = self.attention(query, key, value, input_mask)

        # (B * num_heads) x T x head_dim) -> B x T x C
        out = torch.cat(out.chunk(self.num_heads, dim=0), dim=-1)

        out = self.wo(out)
        
        # B x T x C -> T x B x C
        out = out.permute(1, 0, 2)

        # compatible with the MultiheadAttention class
        attn_weights = None

        return out, attn_weights
