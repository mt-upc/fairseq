import re
import logging
from dataclasses import dataclass, field
from omegaconf import II, DictConfig
from typing import Any, Optional, Dict, List, Type

import torch
import torch.nn as nn

from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import (
    merge_with_parent,
    convert_namespace_to_omegaconf
)
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
)
from fairseq.models.wav2vec import (
    Wav2VecEncoder,
    Wav2Vec2Config,
    Wav2Vec2AsrConfig,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.models.hubert import (
    HubertEncoder,
    HubertConfig,
    HubertAsrConfig,
)
from fairseq.models.transformer import (
    TransformerModelBase,
    TransformerDecoder,
)
from fairseq.modules.adapter import ScaledParallelAdapter
from fairseq.models.speech_to_text import Conv1dSubsampler
from fairseq.tasks import FairseqTask
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.utils import safe_hasattr


logger = logging.getLogger(__name__)


@dataclass
class AdaptersConfig(FairseqDataclass):
    adapter_dim: int = field(
        default=256, metadata={"help": "Bottleneck dimension for the Scaled Parallel Adaoter"}
    )
    adapter_scale: float = field(
        default=1, metadata={"help": "Scaling factor of the parallel adapter output"}
    )
    apply_at_self_attn: bool = field(
        default=False,
        metadata={"help":
            "Apply Prefix Tuning at the Self-Attention sublayers of this component"}
    )
    apply_at_cross_attn: bool = field(
        default=False,
        metadata={"help":
            "Apply Prefix Tuning at the Cross-Attention sublayers of this component"}
    )
    apply_at_ffn: bool = field(
        default=False,
        metadata={"help":
            "Apply Scaled Parallel Adapter at the Feed-forward sublayers of this component"}
    )
    

@dataclass
class S2TLengthAdaptorConfig(FairseqDataclass):
    in_channels: int = field(
        default=1024,
        metadata={"help": "# of input channels in the Length Adaptor"}
    )
    mid_channels: int = field(
        default=1024,
        metadata={"help": "# of intermediate channels in the Length Adaptor"}
    )
    out_channels: int = field(
        default=1024,
        metadata={"help": "# of output channels in the Length Adaptor"}
    )

    kernel_sizes: List[int] = field(
        default_factory=lambda: [3, 3, 3],
        metadata={"help": "kernel size of each Conv1d layer in the Length Adaptor"}
    )


@dataclass
class W2VTimeChannelMaskingConfig(FairseqDataclass):
    length: int = field(
        default=10, metadata={"help": "length of the mask"}
    )
    prob: float = field(
        default=0.0,
        metadata={"help": "probability of masking a token/feature"}
    )
    selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )


@dataclass
class W2VMaskingConfig(FairseqDataclass):
    apply: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    time: W2VTimeChannelMaskingConfig = field(
        default_factory=lambda: W2VTimeChannelMaskingConfig('time'),
        metadata={"help": "time masking configuration"},
    )
    channels: W2VTimeChannelMaskingConfig = field(
        default_factory=lambda: W2VTimeChannelMaskingConfig('channel'),
        metadata={"help": "channel masking configuration"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )
    channel_before: bool = False


@dataclass
class S2TPretrainedComponentConfig(FairseqDataclass):
    path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pretrained component"}
    )
    no_load_weights: bool = field(
        default=False,
        metadata={"help": "don't load weights from pretrained component"},
    )
    layers_to_freeze: List[str] = field(
        default_factory= lambda: [],
        metadata={"help": "list of layers to freeze in pretrained component (no_load_weights must be False)"},
    )
    layers_to_reset: List[str] = field(
        default_factory= lambda: [],
        metadata={"help": "list of layers to reset in pretrained component (no_load_weights must be False)"},
    )
    dropout: float = field(default=0.0, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN."},
    )
    layerdrop: float = field(default=0, metadata={"help": "LayerDrop probability"})
    pre_args: Any = None # Store the args once the training has started
    data: str = II("task.data")


@dataclass
class S2TPretrainedEncoderConfig(S2TPretrainedComponentConfig):
    length_adaptor: Optional[S2TLengthAdaptorConfig] = field(
        default=None,
        metadata={"help": "length adaptor configuration"},
    )

    # Arguments for wav2vec(-ish) encoders
    masking: W2VMaskingConfig = field(
        default_factory=lambda: W2VMaskingConfig(),
        metadata={"help": "masking configuration for wav2vec(-ish) encoders"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    adapters: Optional[AdaptersConfig] = field(
        default=None,
        metadata={"help": "apply adapters to each layer"}
    )


@dataclass
class S2TPretrainedDecoderConfig(S2TPretrainedComponentConfig):
    cross_attention_dropout: float = field(
        default=II('model.decoder.attention_dropout'),
        metadata={"help": "dropout probability for cross-attention weights"},
    )
    adapters: Optional[AdaptersConfig] = field(
        default=None,
        metadata={"help": "apply adapters to each layer"}
    )



@dataclass
class S2TPretrainedConfig(FairseqDataclass):
    encoder: S2TPretrainedEncoderConfig = field(
        default_factory=lambda: S2TPretrainedEncoderConfig('encoder'),
        metadata={"help": "encoder configuration"},
    )
    decoder: S2TPretrainedDecoderConfig = field(
        default_factory=lambda: S2TPretrainedDecoderConfig('decoder'),
        metadata={"help": "decoder configuration"},
    )
    

@register_model("s2t_pretrained", dataclass=S2TPretrainedConfig)
class S2TPretrainedModel(FairseqEncoderDecoderModel):
    """ Model comprising pretrained encoder and decoder. """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: S2TPretrainedConfig, task: SpeechToTextTask) -> "S2TPretrainedModel":
        encoder = S2TPretrainedComponent.build(cfg.encoder)
        decoder = S2TPretrainedComponent.build(cfg.decoder, task.target_dictionary)
        return cls(encoder, decoder)


class S2TPretrainedComponent:
    """ Base class for pretrained S2T encoders and decoders. """
    
    def __init__(self, cfg: S2TPretrainedComponentConfig):
        self.cfg_ = cfg

    @staticmethod
    def load_pre_args(cfg: S2TPretrainedComponentConfig, state: Dict) -> None:
        if state.get("cfg", None) is not None:
            cfg.pre_args = state['cfg']
        elif state.get("args", None) is not None:
            cfg.pre_args = convert_namespace_to_omegaconf(state["args"])
        else:
            raise ValueError('Could not find args in checkpoint')
        
    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedComponentConfig) -> None:
        cfg.pre_args.model.dropout = cfg.dropout
        cfg.pre_args.model.attention_dropout = cfg.attention_dropout
        cfg.pre_args.model.activation_dropout = cfg.activation_dropout

    @classmethod
    def build(cls, cfg: S2TPretrainedComponentConfig, dictionary: Dictionary = None) -> Type['S2TPretrainedComponent']:
        training = not safe_hasattr(cfg, 'pre_args')
        component_type = cfg._name

        if training:
            logger.info(f"Building {component_type} from: {cfg.path}")
            state = load_checkpoint_to_cpu(cfg.path)
            S2TPretrainedComponent.load_pre_args(cfg, state)

        if component_type == 'encoder':
            component = S2TPretrainedEncoder.get_class(cfg).build(cfg)
            if safe_hasattr(cfg, "length_adaptor") and not safe_hasattr(component, "length_adaptor"):
                component.add_length_adaptor(cfg.length_adaptor)
        elif component_type == 'decoder':
            component = S2TPretrainedDecoder.get_class(cfg).build(cfg, dictionary)
        else:
            raise ValueError("Invalid config type")

        if training:
            if cfg.no_load_weights:
                logger.info(f"Not loading weights from pretrained {component_type}")
            else:
                logger.info(f"Loading weights from pretrained {component_type}")
                component.load_weights(state)
                component.freeze_layers()
                
        if safe_hasattr(cfg, "adapters"):
            component.add_adapters(cfg.adapters)

        return component

    def load_weights(self, state: Dict):
        reset_params = []
        for n in list(state['model'].keys()):
            for l in self.cfg_.layers_to_reset:
                l = re.compile(eval(l))
                if re.match(l, n):
                    state['model'].pop(n)
                    reset_params.append(n)
        logger.info(
            f"Parameters to be resetted:\n\t" + '\n\t'.join(reset_params)
        )

        missing_keys, unexpected_keys = \
            self.load_state_dict(state['model'], strict=False)

        logger.info(
            f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + \
                '\n\t'.join(missing_keys)
        )
        logger.info(
            f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys)
        )

    def freeze_layers(self) -> None:
        frozen_params = []
        for n, p in self.named_parameters():
            for l in self.cfg_.layers_to_freeze:
                l = re.compile(eval(l))
                if re.match(l, n):
                    p.requires_grad = False
                    frozen_params.append(n)
        logger.info(
            f"Freezing parameters:\n\t" + '\n\t'.join(frozen_params)
        )


class S2TPretrainedEncoder(FairseqEncoder, S2TPretrainedComponent):
    """ Base class for pretrained S2T encoders """

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        FairseqEncoder.__init__(self, dictionary=None)
        S2TPretrainedComponent.__init__(self, cfg)
        
        self.embed_dim = cfg["pre_args"]["model"]["w2v_args"]["model"].encoder_embed_dim

    @classmethod
    def get_class(cls, cfg: S2TPretrainedEncoderConfig) -> Type['S2TPretrainedEncoder']:
        name = cfg.pre_args.model._name
        if name.startswith('wav2vec'):
            return PretrainedWav2VecEncoder
        elif name.startswith('hubert'):
            return PretrainedHubertEncoder
        else:
            raise ValueError(f"Unknown encoder name: {name}")

    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedEncoderConfig) -> None:
        return S2TPretrainedComponent.update_pre_args(cfg)

    @classmethod
    def build(cls, cfg: S2TPretrainedEncoderConfig) -> 'S2TPretrainedEncoder':
        cls.update_pre_args(cfg)
        return cls(cfg)

    def add_length_adaptor(self, cfg: S2TLengthAdaptorConfig) -> None:
        self.length_adaptor = Conv1dSubsampler(
            in_channels=cfg.in_channels,
            mid_channels=cfg.mid_channels,
            out_channels=cfg.out_channels,
            kernel_sizes=cfg.kernel_sizes,
        )

    def pre_forward(self, src_tokens, src_lengths, **kwargs):
        return {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            **kwargs,
        }

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_inputs = self.pre_forward(src_tokens, src_lengths, **kwargs)
        encoder_out = self.ORIGINAL_MODEL_CLS.forward(self, **encoder_inputs)
        return self.post_forward(encoder_out)

    def post_forward(self, encoder_out):
        if safe_hasattr(self, "length_adaptor"):
            for i, (eo, epm) in enumerate(zip(encoder_out["encoder_out"], encoder_out["encoder_padding_mask"])):
                eo = eo.transpose(0, 1)
                lengths = (~epm).sum(dim=1) \
                    if epm is not None else torch.LongTensor([eo.size(1)] * eo.size(0))
                encoder_out["encoder_out"][i], lengths = self.length_adaptor(eo, lengths.to(eo.device))
                encoder_out["encoder_padding_mask"][i] = lengths_to_padding_mask(lengths)
        return encoder_out
    
    def add_adapters(self, cfg: AdaptersConfig) -> None:
        def make_adapter(self, cfg):
            return ScaledParallelAdapter(
                    embed_dim=self.embed_dim,
                    bottleneck_dim=cfg.adapter_dim,
                    scaling_factor=cfg.adapter_scale
                )
        for transformer_layer in self.w2v_model.encoder.layers:
            if cfg.apply_at_self_attn:
                transformer_layer.self_attn_adapter = make_adapter(self, cfg)
            if cfg.apply_at_ffn:
                transformer_layer.ffn_adapter = make_adapter(self, cfg)

class PretrainedWav2VecBaseEncoder(S2TPretrainedEncoder):
    """ Base class for pretrained wav2vec(-ish) encoders, including HuBERT """

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        super().__init__(cfg)

    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedComponentConfig) -> None:
        super().update_pre_args(cfg)
        cfg.pre_args.model.final_dropout = cfg.final_dropout
        cfg.pre_args.model.w2v_args.model.dropout = cfg.dropout
        cfg.pre_args.model.w2v_args.model.attention_dropout = cfg.attention_dropout
        cfg.pre_args.model.w2v_args.model.activation_dropout = cfg.activation_dropout
        cfg.pre_args.model.w2v_args.model.dropout_input = cfg.dropout_input
        cfg.pre_args.model.w2v_args.model.encoder_layerdrop = cfg.layerdrop

        cfg.pre_args.model.apply_mask = cfg.masking.apply
        cfg.pre_args.model.w2v_args.model.mask_length = cfg.masking.time.length
        cfg.pre_args.model.w2v_args.model.mask_channel_length = cfg.masking.channels.length
        cfg.pre_args.model.w2v_args.model.mask_prob = cfg.masking.time.prob
        cfg.pre_args.model.w2v_args.model.mask_channel_prob = cfg.masking.channels.prob
        cfg.pre_args.model.w2v_args.model.mask_selection = cfg.masking.time.selection
        cfg.pre_args.model.w2v_args.model.mask_channel_selection = cfg.masking.channels.selection
        cfg.pre_args.model.w2v_args.model.mask_other = cfg.masking.time.other
        cfg.pre_args.model.w2v_args.model.mask_channel_other = cfg.masking.channels.other
        cfg.pre_args.model.w2v_args.model.no_mask_overlap = cfg.masking.time.no_overlap
        cfg.pre_args.model.w2v_args.model.no_mask_channel_overlap = cfg.masking.channels.no_overlap
        cfg.pre_args.model.w2v_args.model.mask_min_space = cfg.masking.time.min_space
        cfg.pre_args.model.w2v_args.model.mask_channel_min_space = cfg.masking.channels.min_space
        cfg.pre_args.model.freeze_finetune_updates = 0
        
        try:
            cfg.pre_args.model.w2v_args.model.require_same_masks = cfg.masking.require_same_masks
            cfg.pre_args.model.w2v_args.model.mask_dropout = cfg.masking.dropout
            cfg.pre_args.model.w2v_args.model.mask_channel_before = cfg.masking.channel_before
        except:
            pass

    @classmethod
    def build(cls, cfg: S2TPretrainedEncoderConfig) -> 'PretrainedWav2VecBaseEncoder':
        if cfg.pre_args.model._name == cls.PRETRAIN_MODEL_NAME:
            model_args = DictConfig(
                cls.FINETUNE_MODEL_CFG(cls.FINETUNE_MODEL_NAME)
            )
        elif cfg.pre_args.model._name == cls.FINETUNE_MODEL_NAME:
            model_args = merge_with_parent(
                cls.FINETUNE_MODEL_CFG,
                cfg.pre_args.model,
                remove_missing=True
            )
        else:
            raise ValueError(
                f"Unknown model name: {cfg.pre_args.model._name}"
            )

        if not safe_hasattr(model_args, 'w2v_args'):
            model_args.w2v_args = cfg.pre_args
        
        model_args.w2v_args.model = merge_with_parent(
            cls.PRETRAIN_MODEL_CFG,
            model_args.w2v_args.model,
            remove_missing=True,
        )
        model_args.w2v_args.task = merge_with_parent(
            cls.PRETRAIN_TASK_CFG,
            model_args.w2v_args.task,
            remove_missing=True,
        )
        model_args.data = cfg.data
        model_args.normalize = model_args.w2v_args.task.normalize

        cfg.pre_args.model = model_args

        return super().build(cfg)

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('w2v_model.'):
                if k.startswith('w2v_encoder.'):            # Finetuned models
                    k = re.sub(r'^w2v_encoder.', '', k)
                else:                                       # Pretrained models
                    k = f"w2v_model.{k}"
            if k.endswith('label_embs_concat') or k.startswith('proj.'):
                continue                                    # This layers are not used
            new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, strict=strict)

    def pre_forward(self, src_tokens, src_lengths, **kwargs):
        encoder_in = super().pre_forward(src_tokens, src_lengths, **kwargs)
        encoder_in['source'] = encoder_in.pop('src_tokens')
        encoder_in['padding_mask'] = lengths_to_padding_mask(src_lengths)
        return encoder_in

    def post_forward(self, encoder_out):
        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        encoder_out = {k: [v] for k, v in encoder_out.items()}
        super().post_forward(encoder_out)
        return encoder_out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        for i, (eo, epm) in enumerate(zip(
            encoder_out["encoder_out"],
            encoder_out["encoder_padding_mask"],
        )):
            if eo is not None:
                encoder_out["encoder_out"][i] = \
                    eo.index_select(1, new_order)
            if epm is not None:
                encoder_out["encoder_padding_mask"][i] = \
                    epm.index_select(0, new_order)

        return encoder_out


class PretrainedWav2VecEncoder(PretrainedWav2VecBaseEncoder, Wav2VecEncoder):
    """ Pretrained wav2vec encoder """

    PRETRAIN_MODEL_NAME = 'wav2vec2'
    PRETRAIN_MODEL_CFG = Wav2Vec2Config
    PRETRAIN_TASK_CFG = AudioPretrainingConfig
    FINETUNE_MODEL_NAME = 'wav2vec_ctc'
    FINETUNE_MODEL_CFG = Wav2Vec2AsrConfig
    ORIGINAL_MODEL_CLS = Wav2VecEncoder

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        PretrainedWav2VecBaseEncoder.__init__(self, cfg)
        Wav2VecEncoder.__init__(self, cfg.pre_args.model)


class HubertDummyTask(FairseqTask):
    """ Dummy task for HuBERT """

    def state_dict(self):
        return {'dictionaries': [None]}

    @property
    def target_dictionary(self):
        return None


class PretrainedHubertEncoder(PretrainedWav2VecBaseEncoder, HubertEncoder):
    """ Pretrained HuBERT encoder """

    PRETRAIN_MODEL_NAME = 'hubert'
    PRETRAIN_MODEL_CFG = HubertConfig
    PRETRAIN_TASK_CFG = HubertPretrainingConfig
    FINETUNE_MODEL_NAME = 'hubert_ctc'
    FINETUNE_MODEL_CFG = HubertAsrConfig
    ORIGINAL_MODEL_CLS = HubertEncoder

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        PretrainedWav2VecBaseEncoder.__init__(self, cfg)
        HubertEncoder.__init__(self, cfg.pre_args.model, HubertDummyTask(None))


class S2TPretrainedDecoder(FairseqDecoder, S2TPretrainedComponent):
    """ Base class for pretrained S2T decoders """

    def __init__(self, cfg: S2TPretrainedDecoderConfig, tgt_dict: Dictionary):
        FairseqDecoder.__init__(self, dictionary=tgt_dict)
        S2TPretrainedComponent.__init__(self, cfg)
        
        self.embed_dim = cfg["pre_args"]["model"].decoder_embed_dim

    @classmethod
    def get_class(cls, cfg: S2TPretrainedDecoderConfig) -> Type['S2TPretrainedEncoder']:
        name = cfg.pre_args.model._name
        if name.startswith('mbart'):
            return PretrainedBartDecoder
        else:
            raise ValueError(f"Unknown decoder name: {name}")

    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedDecoderConfig) -> None:
        return S2TPretrainedComponent.update_pre_args(cfg)

    @classmethod
    def build(cls, cfg: S2TPretrainedDecoderConfig, tgt_dict: Dictionary) -> 'S2TPretrainedDecoder':
        cls.update_pre_args(cfg)
        return cls(cfg, tgt_dict)
    
    def add_adapters(self, cfg: AdaptersConfig) -> None:       
        def make_adapter(self, cfg):
            return ScaledParallelAdapter(
                    embed_dim=self.embed_dim,
                    bottleneck_dim=cfg.adapter_dim,
                    scaling_factor=cfg.adapter_scale
                )
        for transformer_layer in self.layers:
            if cfg.apply_at_self_attn:
                transformer_layer.self_attn_adapter = make_adapter(self, cfg)
            if cfg.apply_at_cross_attn:
                transformer_layer.cross_attn_adapter = make_adapter(self, cfg)
            if cfg.apply_at_ffn:
                transformer_layer.ffn_adapter = make_adapter(self, cfg)

class PretrainedBartDecoder(S2TPretrainedDecoder, TransformerDecoder):
    """ Pretrained BART decoder """

    def __init__(self, cfg: S2TPretrainedDecoderConfig, tgt_dict: Dictionary, embed_tokens: nn.Embedding):
        S2TPretrainedDecoder.__init__(self, cfg, tgt_dict)
        TransformerDecoder.__init__(self, cfg.pre_args.model, tgt_dict, embed_tokens)

    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedComponentConfig) -> None:
        super().update_pre_args(cfg)
        cfg.pre_args.model.layerdrop = cfg.layerdrop

    @classmethod
    def build(cls, cfg: S2TPretrainedDecoderConfig, tgt_dict: Dictionary) -> 'S2TPretrainedDecoder':
        cls.update_pre_args(cfg)
        embed_tokens = TransformerModelBase.build_embedding(
            cfg.pre_args.model,
            tgt_dict,
            cfg.pre_args.model.decoder_embed_dim
        )
        decoder = cls(cfg, tgt_dict, embed_tokens)

        # XXX: Put this in S2TPretrainedDecoder class in the future
        if cfg.cross_attention_dropout != cfg.attention_dropout:
            for l in decoder.layers:
                l.encoder_attn.dropout_module.p = cfg.cross_attention_dropout

        return decoder

    def load_state_dict(self, state_model, strict=True):
        new_state_dict = {}
        for k, v in state_model.items():
            if k.startswith('encoder.'):
                continue
            k = re.sub(r'^decoder.', '', k)
            new_state_dict[k] = v

        if "output_projection.weight" not in new_state_dict.keys():
            # These layers may be shared
            new_state_dict["output_projection.weight"] = \
                new_state_dict["embed_tokens.weight"]

        return S2TPretrainedDecoder.load_state_dict(self, new_state_dict, strict=strict)
