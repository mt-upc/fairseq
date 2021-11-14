import logging
from pathlib import Path
from typing import List
from local_attention.local_attention import default

import torch.nn as nn

from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
)
from fairseq.models.speech_to_text.utils import parse_str2tuple
from fairseq.modules import MultiformerEncoderLayer

logger = logging.getLogger(__name__)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
        strides: List[int] = (2, 2),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)
        self.strides = strides
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                s,
                padding=k // 2,
            )
            for i, (k, s) in enumerate(zip(kernel_sizes, strides))
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for s in self.strides:
            out = ((out.float() - 1) / s + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        # B x T x (C x D) -> B x (C x D) x T
        x = src_tokens.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)

        # B x (C x D) x T -> T x B x (C x D)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("s2t_multiformer")
class S2TMultiformerModel(S2TTransformerModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        parser.add_argument(
            "--conv-strides",
            type=str,
            metavar="N",
            help="strides of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--full-attention-heads",
            type=int,
            metavar="N",
            help="Number of full attention heads",
        )
        parser.add_argument(
            "--local-att-nheads",
            type=str,
            metavar="N",
            help="Local Attention number of heads of each group",
        )
        parser.add_argument(
            "--local-att-ws",
            type=str,
            metavar="N",
            help="Local Attention window size of each group",
        )
        parser.add_argument(
            "--compressed-att-nheads",
            type=str,
            metavar="N",
            help="Compressed Attention number of heads of each group",
        )
        parser.add_argument(
            "--compressed-att-ks",
            type=str,
            metavar="N",
            help="Compressed Attention kernel size of each group",
        )
        parser.add_argument(
            "--compressed-att-cf",
            type=str,
            metavar="N",
            help="Compressed Attention compression factor of each group",
        )
        parser.add_argument(
            "--compressed-att-conv-type",
            type=str,
            metavar="N",
            help="Convolutions type ('normal', 'separable' or 'depthwise')",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TMultiformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder
    

class S2TMultiformerEncoder(S2TTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        conv_kernel_sizes = parse_str2tuple(args.conv_kernel_sizes)
        conv_strides = parse_str2tuple(args.conv_strides)
        assert len(conv_kernel_sizes) == len(conv_strides)
        
        if len(conv_kernel_sizes) > 0:
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                conv_kernel_sizes,
                conv_strides,
            )
        else:
            self.subsample = nn.Identity()

        full_att_heads = args.full_att_heads
        local_att_cfg = self.build_local_att_cfg(args)
        compressed_att_cfg = self.build_compressed_att_cfg(args)
        compressed_conv_type = args.compressed_att_conv_type

        self.transformer_layers = nn.ModuleList(
            [MultiformerEncoderLayer(args, full_att_heads, local_att_cfg, compressed_att_cfg, compressed_conv_type) for _ in range(args.encoder_layers)]
        )

    def build_local_att_cfg(self, args):
        local_att_nheads = parse_str2tuple(args.local_att_nheads)
        local_att_ws = parse_str2tuple(args.local_att_ws)
        assert len(local_att_nheads) == len(local_att_ws)
        local_att_cfg = tuple(zip(local_att_nheads, local_att_ws))
        return local_att_cfg
    
    def build_compressed_att_cfg(self, args):
        compressed_att_nheads = parse_str2tuple(args.compressed_att_nheads)
        compressed_att_ks = parse_str2tuple(args.compressed_att_ks)
        compressed_att_cf = parse_str2tuple(args.compressed_att_cf)
        assert len(compressed_att_nheads) == len(compressed_att_ks) == len(compressed_att_cf)
        compressed_att_cfg = tuple(zip(compressed_att_nheads, compressed_att_ks, compressed_att_cf))
        return compressed_att_cfg


@register_model_architecture(model_name="s2t_multiformer", arch_name="s2t_multiformer")
def base_architecture(args):
    from fairseq.models.speech_to_text.s2t_transformer import base_architecture
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "")
    args.conv_strides = getattr(args, "conv_strides", "")
    args.full_att_heads = getattr(args, "full_att_heads", 0)
    args.local_att_nheads = getattr(args, "local_att_nheads", "4")
    args.local_att_ws = getattr(args, "local_att_ws", "64")
    args.compressed_att_nheads = getattr(args, "compressed_att_nheads", "4")
    args.compressed_att_ks = getattr(args, "compressed_att_ks", "9")
    args.compressed_att_cf = getattr(args, "compressed_att_cf", "4")
    args.compressed_att_conv_type = getattr(args, "compressed_att_conv_type", "depthwise")
    base_architecture(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_s")
def s2t_local_transformer_s(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_s
    args.local_att_nheads = getattr(args, "local_att_nheads", "2")
    args.compressed_att_nheads = getattr(args, "compressed_att_nheads", "2")
    s2t_transformer_s(args)
    base_architecture(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_xs")
def s2t_local_transformer_xs(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_xs
    s2t_transformer_xs(args)
    s2t_local_transformer_s(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_sp")
def s2t_local_transformer_sp(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_sp
    s2t_transformer_sp(args)
    s2t_local_transformer_s(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_m")
def s2t_local_transformer_m(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_m
    args.local_att_nheads = getattr(args, "local_att_nheads", "4")
    args.compressed_att_nheads = getattr(args, "compressed_att_nheads", "4")
    s2t_transformer_m(args)
    base_architecture(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_mp")
def s2t_local_transformer_mp(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_mp
    s2t_transformer_mp(args)
    s2t_local_transformer_m(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_l")
def s2t_local_transformer_l(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_l
    args.local_att_nheads = getattr(args, "local_att_nheads", "8")
    args.compressed_att_nheads = getattr(args, "compressed_att_nheads", "8")
    s2t_transformer_l(args)
    base_architecture(args)


@register_model_architecture("s2t_multiformer", "s2t_multiformer_lp")
def s2t_local_transformer_lp(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_lp
    s2t_transformer_lp(args)
    s2t_local_transformer_l(args)
