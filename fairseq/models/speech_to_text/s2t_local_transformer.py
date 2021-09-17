import logging
from pathlib import Path

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
from fairseq.modules import LocalTransformerEncoderLayer

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
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / self.stride + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("s2t_local_transformer")
class S2TLocalTransformerModel(S2TTransformerModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        super().add_args(parser)
        parser.add_argument(
            "--conv-strides",
            type=str,
            metavar="N",
            help="strides of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--encoder-attention-window",
            type=str,
            metavar="N",
            help="local attention window size",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TLocalTransformerEncoder(args)
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
    

class S2TLocalTransformerEncoder(S2TTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)
        attention_windows = eval(args.attention_window)

        if isinstance(attention_windows, int):
            attention_windows = [attention_windows] * args.encoder_layers
        elif isinstance(attention_windows, tuple):
            attention_windows = list(attention_windows)
            assert len(attention_windows) == args.encoder_layers
        else:
            raise TypeError("'attention_window' format unknown")

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
            [int(s) for s in args.conv_strides.split(",")],
        )
        self.transformer_layers = nn.ModuleList(
            [LocalTransformerEncoderLayer(args, w) for w in attention_windows]
        )


@register_model_architecture(model_name="s2t_local_transformer", arch_name="s2t_local_transformer")
def base_architecture(args):
    from fairseq.models.speech_to_text.s2t_transformer import base_architecture
    base_architecture(args)
    args.conv_strides = getattr(args, "conv_strides", "2,2")


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_s")
def s2t_local_transformer_s(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_s
    s2t_transformer_s(args)
    base_architecture(args)


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_xs")
def s2t_local_transformer_xs(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_xs
    s2t_transformer_xs(args)
    s2t_local_transformer_s(args)


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_sp")
def s2t_local_transformer_sp(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_sp
    s2t_transformer_sp(args)
    s2t_local_transformer_s(args)


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_m")
def s2t_local_transformer_m(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_m
    s2t_transformer_m(args)
    base_architecture(args)


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_mp")
def s2t_local_transformer_mp(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_mp
    s2t_transformer_mp(args)
    s2t_local_transformer_m(args)


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_l")
def s2t_local_transformer_l(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_l
    s2t_transformer_l(args)
    base_architecture(args)


@register_model_architecture("s2t_local_transformer", "s2t_local_transformer_lp")
def s2t_local_transformer_lp(args):
    from fairseq.models.speech_to_text.s2t_transformer import s2t_transformer_lp
    s2t_transformer_lp(args)
    s2t_local_transformer_l(args)
