#!/usr/bin/env python3

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention_layer import CrossAttentionLayer
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import FairseqEncoder
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)


class S2TPerceiverEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.num_latents = args.num_latents
        self.dla_train_num_latents = args.dla_train_num_latents
        self.dla_inf_num_latents = args.dla_inf_num_latents
        self.latent_dim = args.encoder_embed_dim

        # initialize latent vectors
        self.init_latent = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_latents, 1, self.latent_dim),
                mean=0,
                std=0.05,
            )
        )

        # input processor
        self.processor_dropout = FairseqDropout(
            p=args.conv_dropout, module_name=self.__class__.__name__
        )
        if not args.no_conv:
            self.conv_processor = Conv1dSubsampler(
                args.input_feat_per_channel,
                args.conv_channels,
                args.conv_output_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
                args.conv_stride,
            )
            self.input_dim = args.conv_output_dim
        else:
            self.input_dim = args.input_feat_per_channel

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.input_dim)

        # positional embeddings
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, self.input_dim, 1
        )

        # cross attention layer
        self.cross_attention_layer = CrossAttentionLayer(
            self.latent_dim,
            args.encoder_ffn_embed_dim,
            self.input_dim,
            args.activation_fn
        )

        # latent transformer
        self.self_attention_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        self.final_layer_norm = LayerNorm(self.latent_dim)

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):

        encoder_states = []

        if hasattr(self, "conv_processor"):
            src_processed, src_lengths = self.conv_processor(src_tokens, src_lengths)
        else:
            src_processed = src_tokens.transpose(0, 1)

        src_processed = src_processed * self.embed_scale

        src_mask = lengths_to_padding_mask(src_lengths)
        pos = self.embed_positions(src_mask).transpose(0, 1)
        src_processed += pos

        src_processed = self.processor_dropout(src_processed)

        latents = self.init_latent.expand(
            self.num_latents, src_processed.size(1), self.latent_dim
        ).to(device=src_processed.device)
        
        if self.training:
            latents = self.dla_train_selection(latents)

        # cross attention
        latents, cros_attn_weights = self.cross_attention_layer(latents, src_processed, src_mask)
        
        if not self.training:
            latents = self.dla_inf_selection(latents, cros_attn_weights)

        # latent attention
        for self_attention_layer in self.self_attention_layers:
            latents = self_attention_layer(latents, encoder_padding_mask=None)
            if return_all_hiddens:
                encoder_states.append(latents)

        latents = self.final_layer_norm(latents)

        return {
            "encoder_out": [latents],
            "encoder_padding_mask": [],
            "encoder_embedding": [],
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": []
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )
        return x
    
    def dla_train_selection(self, latents):
        
        if self.dla_train_num_latents == self.num_latents:
            return latents
        
        bs = latents.size(1)
        
        dla_train_idx = np.empty(shape=[self.dla_train_num_latents, bs])
        for i in range(bs):
            dla_train_idx[:, i] = np.random.choice(
                self.num_latents, size=self.dla_train_num_latents, replace=False
            )
        dla_train_idx = torch.tensor(dla_train_idx).to(
            dtype=torch.long, device=latents.device
        )
        dla_train_idx = dla_train_idx.unsqueeze(-1).expand(
            self.dla_train_num_latents, bs, self.latent_dim
        )

        return torch.gather(latents, dim=0, index=dla_train_idx)
    
    def dla_inf_selection(self, latents, alpha):
        
        if self.dla_inf_num_latents == self.num_latents:
            return latents
        
        bs = alpha_norm.size(0)
        
        # L2-normalize
        alpha_norm = F.normalize(alpha, p=2, dim=2)
        
        # absolute cosine similarity matrix
        s = torch.bmm(alpha_norm, alpha_norm.transpose(1, 2))
        s = torch.abs(s)
        
        # easy index
        arng = torch.arange(self.num_latents)
        arng_bs = torch.arange(bs).unsqueeze(-1)
        
        # mask diagonal
        s[:, arng, arng] = -1.0
        
        # select first
        max_s = torch.max(s, dim=2)[0]
        dla_inf_indices = max_s.argmin(dim=1).unsqueeze(1)
        
        # iterative select the rest k'-1
        while dla_inf_indices.size(1) < self.dla_inf_num_latents:
            max_s = s[arng_bs, dla_inf_indices].max(dim=1)[0] 
            max_s[arng_bs, dla_inf_indices] = 999_999
            dla_inf_indices = torch.cat(
                [dla_inf_indices, max_s.argmin(dim=1).unsqueeze(1)], dim=1
            )
                    
        return torch.gather(latents, dim=0, index=dla_inf_indices)

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # list[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
