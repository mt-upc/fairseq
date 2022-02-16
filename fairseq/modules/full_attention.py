import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(self, query, key, value, mask=None):
        """ Computes the Scaled Dot-Product Attention

        Args:
            query (torch.FloatTensor): Query Tensor  (... x T_q x d_q)
            key (torch.FloatTensor): Key Tensor      (... x T_k x d_k)
            value (torch.FloatTensor): Value Tensor  (... x T_v x d_v)
            mask (torch.BoolTensor): Attention mask  (... x T_q x T_k)

        Returns:
            torch.FloatTensor: Result of the SDPA    (... x T_q x d_v)
            torch.FloatTensor: Attention map         (... x T_q x T_k)

        """

        attn_logits = torch.matmul(query, key.transpose(-2, -1))

        attn_logits = attn_logits / math.sqrt(key.size(-1))

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attention = F.softmax(attn_logits.float(), dim=-1)
        attention = attention.half() if attn_logits.dtype is torch.float16 else attention
        attention = self.dropout(attention)

        output = torch.matmul(attention, value)
        return output, attention

    def forward(self, query, key, value, key_padding_mask=None):
        """ Computes the Scaled Dot-Product Attention

        Args:
            query (torch.FloatTensor):  Query Tensor                      (... x T_q x d_q)
            key (torch.FloatTensor):  Key Tensor                          (... x T_k x d_k)
            value (torch.FloatTensor):  Value Tensor                      (... x T_v x d_v)
            key_padding_mask (torch.BoolTensor): Attention mask of keys   (... x T_k)

        Returns:
            torch.FloatTensor: Result of the Global Attention    (... x T_q x d_v)

        """

        assert key.size(-1) == value.size(-1), "Key and Value dimensions must coincide"

        if key_padding_mask is None:
            key_padding_mask = torch.zeros_like(key[..., 0]).bool().to(key.device)
        attn_mask = key_padding_mask.unsqueeze(-2).expand(-1, query.size(-2), -1)

        out, _ = self.scaled_dot_product(query, key, value, attn_mask)

        return out
