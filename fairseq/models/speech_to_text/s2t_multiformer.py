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
            "--conv-ks",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-strides",
            type=str,
            metavar="N",
            help="strides of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--arg-supremo",
            type=str,
            metavar="N",
            help="Multiformer encoder layers configuration",
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

        conv_kernel_sizes = parse_str2tuple(args.conv_ks)
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
            self.subsample = Identity2D()

        arg_supremo = eval(args.arg_supremo)
        layer_configs = []
        for layer_config in arg_supremo:
            layer = {}
            layer['full_att_heads'] = self.build_config_full(layer_config)
            layer['fast_att_heads'] = self.build_config_fast(layer_config)
            layer['local_att_cfg'] = self.build_config_local(layer_config)
            layer['compressed_att_cfg'] = self.build_config_compressed(layer_config)
            layer_configs.append(layer)

        self.transformer_layers = nn.ModuleList(
            [MultiformerEncoderLayer(args, config['full_att_heads'], config['fast_att_heads'], config['local_att_cfg'], config['compressed_att_cfg']) for config in layer_configs]
        )


    def build_config_full(self, layer_config):
        for group in layer_config:
            if 'full' in group:
                return group[1]
        return 0

    def build_config_fast(self, layer_config):
        for group in layer_config:
            if 'fast' in group:
                return group[1]
        return 0

    def build_config_local(self, layer_config):
        local_heads_config = []
        for group in layer_config:
            if 'local' in group:
                local_heads_config.append((group[1], group[2]))
        return tuple(local_heads_config)

    def build_config_compressed(self, layer_config):
        compressed_heads_config = []
        for group in layer_config:
            if 'compressed' in group:
                compressed_heads_config.append((group[1], group[2], group[3], group[4]))
        return tuple(compressed_heads_config)


class Identity2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_tokens, src_lengths):
        src_tokens = src_tokens.transpose(0, 1).contiguous()
        return src_tokens, src_lengths


@register_model_architecture(model_name="s2t_multiformer", arch_name="s2t_multiformer")
def base_architecture(args):
    from fairseq.models.speech_to_text.s2t_transformer import base_architecture
    args.conv_ks = getattr(args, "conv_ks", "")
    args.conv_strides = getattr(args, "conv_strides", "")
    args.arg_supremo = getattr(args, "arg_supremo", "8 * ((('full', 2), ('fast', 2), ('local', 1, 64), ('local', 1, 128), ('compressed', 2, 9, 5, 'depthwise')),) + 4 * ((('fast', 4), ('com', 4, 64)),)")
    base_architecture(args)
    delattr(args, 'encoder_attention_heads')

@register_model_architecture("s2t_multiformer", "s2t_multiformer_s_h_lc")
def s2t_multiformer_s_h_lc(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_s
    args.conv_ks = getattr(args, "conv_ks", "5,5")
    args.conv_strides = getattr(args, "conv_strides", "2,2")
    args.arg_supremo = getattr(args, "arg_supremo", "12 * ((('local', 2, 64), ('compressed', 2, 5, 2, 'depthwise')),)")
    s2t_transformer_s(args)
    base_architecture(args)

@register_model_architecture("s2t_multiformer", "s2t_multiformer_s")
def s2t_multiformer_s(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_s
    args.conv_ks = getattr(args, "conv_ks", "5,5")
    args.conv_strides = getattr(args, "conv_strides", "2,2")
    args.arg_supremo = getattr(args, "arg_supremo", "2 * ((('compressed', 4, 5, 2, 'depthwise'),),) + 6 * ((('local', 2, 64), ('compressed', 2, 5, 2, 'depthwise')),) + 4 * ((('full', 2,), ('compressed', 2, 7, 3, 'depthwise')),)")
    s2t_transformer_s(args)
    base_architecture(args)

@register_model_architecture("s2t_multiformer", "s2t_multiformer_s_v2")
def s2t_multiformer_s_v2(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_s
    args.conv_ks = getattr(args, "conv_ks", "5,5")
    args.conv_strides = getattr(args, "conv_strides", "2,2")
    args.arg_supremo = getattr(args, "arg_supremo", "6 * ((('local', 1, 64), ('compressed', 3, 5, 2, 'depthwise')),) + 6 * ((('local', 2, 64), ('compressed', 2, 5, 2, 'depthwise')),)")
    s2t_transformer_s(args)
    base_architecture(args)

@register_model_architecture("s2t_multiformer", "s2t_multiformer_s_v5")
def s2t_multiformer_s_v5(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_s
    args.conv_ks = getattr(args, "conv_ks", "5,5")
    args.conv_strides = getattr(args, "conv_strides", "2,2")
    args.arg_supremo = getattr(args, "arg_supremo", "4 * ((('compressed', 4, 5, 2, 'depthwise'),),) + 3 * ((('local', 1, 64), ('compressed', 3, 5, 2, 'depthwise')),) + 5 * ((('local', 2, 64), ('compressed', 2, 5, 2, 'depthwise')),)")
    s2t_transformer_s(args)
    base_architecture(args)
