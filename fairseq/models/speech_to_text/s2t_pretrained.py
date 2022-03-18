import re
import logging
from dataclasses import dataclass, field
from omegaconf import II, DictConfig
from typing import Any, Optional, Dict, Type

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
from fairseq.models.hubert import (
    HubertEncoder,
    HubertConfig,
    HubertAsrConfig,
)
from fairseq.models.transformer import (
    TransformerModelBase,
    TransformerDecoder,
)
from fairseq.tasks import FairseqTask
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.utils import safe_hasattr


logger = logging.getLogger(__name__)


@dataclass
class S2TCouplingLayersConfig(FairseqDataclass):
    in_dim: int = field(
        default=1024, metadata={"help": "input dimension"}
    )
    out_dim: int = field(
        default=1024, metadata={"help": "output dimension"}
    )


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
    coupling: Optional[S2TCouplingLayersConfig] = field(
        default=None,
        metadata={"help": "apply coupling layers to the encoder output"},
    )


@dataclass
class S2TPretrainedDecoderConfig(S2TPretrainedComponentConfig):
    pass


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
    def load_args(cfg: S2TPretrainedComponentConfig, state: Dict) -> None:
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
            S2TPretrainedComponent.load_args(cfg, state)

        if component_type == 'encoder':
            component = S2TPretrainedEncoder.get_class(cfg).build(cfg)
            if safe_hasattr(cfg, "coupling") and not safe_hasattr(component, "coupling_layers"):
                component.add_coupling_layers(cfg.coupling)
        elif component_type == 'decoder':
            component = S2TPretrainedDecoder.get_class(cfg).build(cfg, dictionary)
        else:
            raise ValueError("Invalid config type")

        if training:
            if cfg.no_load_weights:
                logger.info(f"Not loading weights from pretrained {component_type}")
            else:
                logger.info(f"Loading weights from pretrained {component_type}")
                component.load_state_dict(state['model'])

        return component


class S2TPretrainedEncoder(FairseqEncoder, S2TPretrainedComponent):
    """ Base class for pretrained S2T encoders """

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        FairseqEncoder.__init__(self, dictionary=None)
        S2TPretrainedComponent.__init__(self, cfg)

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
        if safe_hasattr(self, "coupling_layers"):
            for i, eo in enumerate(encoder_out["encoder_out"]):
                encoder_out["encoder_out"][i] = self.coupling_layers(eo)
        return encoder_out

    def add_coupling_layers(self, cfg: S2TCouplingLayersConfig) -> None:
        self.coupling_layers = nn.Linear(cfg.in_dim, cfg.out_dim)

    def load_state_dict(self, state_dict, strict=True):
        # Trick to avoid errors when coupling layers are not present in state_dict
        for n, p in self.named_parameters():
            if n.startswith('coupling_layers') and n not in state_dict.keys():
                state_dict[n] = p.data
        super().load_state_dict(state_dict, strict=strict)


class PretrainedWav2VecBaseEncoder(S2TPretrainedEncoder):
    """ Base class for pretrained wav2vec(-ish) encoders, including HuBERT """

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        super().__init__(cfg)

    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedComponentConfig) -> None:
        super().update_pre_args(cfg)
        cfg.pre_args.model.final_dropout = cfg.dropout
        cfg.pre_args.model.w2v_args.model.dropout = cfg.dropout
        cfg.pre_args.model.w2v_args.model.attention_dropout = cfg.attention_dropout
        cfg.pre_args.model.w2v_args.model.activation_dropout = cfg.activation_dropout
        cfg.pre_args.model.w2v_args.model.dropout_input = cfg.dropout
        cfg.pre_args.model.w2v_args.model.encoder_layerdrop = cfg.layerdrop

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

        super().load_state_dict(new_state_dict, strict=strict)

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
        return cls(cfg, tgt_dict, embed_tokens)

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

        S2TPretrainedDecoder.load_state_dict(self, new_state_dict, strict=strict)
