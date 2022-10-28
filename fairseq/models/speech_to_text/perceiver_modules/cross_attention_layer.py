from typing import Tuple

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout


class CrossAttentionLayer(nn.Module):
    """
    Implementation of the Perceiver Cross-Attention Layer
    (https://arxiv.org/abs/2103.03206)
    """

    def __init__(
        self,
        latent_dim,
        ffn_embed_dim,
        input_channels,
        activation_fn,
        dropout=0,
        activation_dropout=0,
        attention_dropout=0,
    ):
        super().__init__()

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=activation_fn)
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        self.input_layer_norm = LayerNorm(input_channels)
        self.attn_layer_norm = LayerNorm(latent_dim)

        self.attn = MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=1,
            kdim=input_channels,
            vdim=input_channels,
            dropout=attention_dropout,
            encoder_decoder_attention=True,
        )

        self.fc1 = nn.Linear(latent_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, latent_dim)

        self.final_layer_norm = LayerNorm(latent_dim)

    def forward(
        self,
        latent_array: torch.FloatTensor,
        input_array: torch.FloatTensor,
        input_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            latent_array: functions as the query for the attention
                [num_latents, batch, latent_dim]
            input_array: functions as the key and values for the attention
                [seq_len, batch, latent_dim]
            input_mask: the mask of the input array (padding)
                [batch, seq_len]
        Returns:
            processed latent_array [num_latents, batch, latent_dim]
            attention weights
        """

        residual = latent_array

        # initial layer norm for the latent and the input
        input_array = self.input_layer_norm(input_array)
        latent_array = self.attn_layer_norm(latent_array)

        latent_array, attn_weights = self.attn(
            query=latent_array,
            key=input_array,
            value=input_array,
            key_padding_mask=input_mask,
        )
        latent_array = self.dropout_module(latent_array)

        latent_array += residual

        latent_array = self.final_layer_norm(latent_array)

        residual = latent_array

        # feed-forward network
        latent_array = self.activation_fn(self.fc1(latent_array))
        latent_array = self.activation_dropout_module(latent_array)
        latent_array = self.fc2(latent_array)

        latent_array = self.dropout_module(latent_array)
        latent_array += residual

        return latent_array, attn_weights