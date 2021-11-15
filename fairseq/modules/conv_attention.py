import torch.nn as nn

from fairseq.modules import FullAttention
from fairseq.data.data_utils import lengths_to_padding_mask


class ConvAttention(nn.Module):
    def __init__(self, dim, kernel_size, stride, conv_type='normal', dropout=0.):
        """ ConvAttention proposed in Speechformer paper
            https://arxiv.org/pdf/2109.04574.pdf
        """
        super().__init__()

        assert conv_type in ['normal', 'separable', 'depthwise']

        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_type = conv_type

        self.conv_k = self.build_convolutional_layer()
        self.conv_v = self.build_convolutional_layer()
        self.attention = FullAttention(dropout)

    def build_convolutional_layer(self):
        convs = []
        convs.append(
            nn.Conv1d(
                in_channels=self.dim,
                out_channels=self.dim,
                groups=(1 if self.conv_type == 'normal' else self.dim),
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.stride//2,
            )
        )

        if self.conv_type == 'separable':
            convs.append(
                nn.Conv1d(
                    in_channels=self.dim,
                    out_channels=self.dim,
                    kernel_size=1,
                )
            )

        return nn.Sequential(*convs) if len(convs) > 1 else convs[0]

    def conv_compression(self, x, conv):
        x = x.permute(0, 2, 1)  # B x T x C -> B x C x T
        x = conv(x)
        x = x.permute(0, 2, 1)  # B x C x T -> B x T x C
        return x

    def get_out_padding_mask(self, in_padding_mask):
        in_lens = (~in_padding_mask).sum(1).clone()
        out_lens = (in_lens + 2 * self.stride//2  - self.kernel_size) \
                    / self.stride + 1
        out_lens = out_lens.floor().int()
        out_padding_mask = lengths_to_padding_mask(out_lens)
        return out_padding_mask

    def forward(self, query, key, value, mask_q=None, mask_kv=None):
        """ Computes the ConvAttention

        Args:
            query (torch.FloatTensor):  Query Tensor (B x T_q x d_q)
            key (torch.FloatTensor):  Key Tensor (B x T_k x d_k)
            value (torch.FloatTensor):  Value Tensor (B x T_v x d_v)
            mask_q (torch.BoolTensor): Attention mask of queries (B x T_q)
            mask_kv (torch.BoolTensor): Attention mask of keys (B x T_k)

        Returns:
            torch.FloatTensor: Result of the Global Attention (B x T_q x d_v)

        """

        key = self.conv_compression(key, self.conv_k)
        value = self.conv_compression(value, self.conv_v)

        mask_kv = self.get_out_padding_mask(mask_kv)

        out = self.attention(query, key, value, mask_q, mask_kv)

        return out
