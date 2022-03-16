import re
import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from omegaconf import II, MISSING, DictConfig

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
from fairseq.utils import safe_getattr, safe_hasattr


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
class S2TPretrainedEncoderConfig(FairseqDataclass):
    path: str = field(
        default=MISSING, metadata={"help": "path to pretrained encoder"}
    )
    args: Any = None # Store the args once the training has started

    coupling: Optional[S2TCouplingLayersConfig] = field(
        default=None,
        metadata={"help": "apply coupling layers to the encoder output"},
    )


@dataclass
class S2TPretrainedDecoderConfig(FairseqDataclass):
    path: str = field(
        default=MISSING, metadata={"help": "path to pretrained decoder"}
    )
    args: Any = None # Store the args once the training has started


@dataclass
class S2TPretrainedConfig(FairseqDataclass):
    encoder: S2TPretrainedEncoderConfig = field(
        default_factory=lambda: S2TPretrainedEncoderConfig(),
        metadata={"help": "encoder configuration"},
    )
    decoder: S2TPretrainedDecoderConfig = field(
        default_factory=lambda: S2TPretrainedDecoderConfig(),
        metadata={"help": "decoder configuration"},
    )
    data: str = II("task.data")
    

@register_model("s2t_pretrained", dataclass=S2TPretrainedConfig)
class S2TPretrainedModel(FairseqEncoderDecoderModel):
    """ Model comprising pretrained encoder and decoder. """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: S2TPretrainedConfig, task: SpeechToTextTask):
        encoder = cls.build_encoder(cfg)
        if safe_hasattr(cfg.encoder, "coupling"):
            encoder.add_coupling_layers(cfg.encoder.coupling)
        decoder = cls.build_decoder(cfg, tgt_dict=task.target_dictionary)
        return S2TPretrainedModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg: S2TPretrainedConfig):
        if safe_getattr(cfg.encoder, 'args') is None:
            state = load_checkpoint_to_cpu(cfg.encoder.path)
            if state.get("cfg", None) is not None:
                cfg.encoder.args = state['cfg']
            elif state.get("args", None) is not None:
                cfg.encoder.args = convert_namespace_to_omegaconf(state["args"])
            else:
                raise ValueError('Could not find args in checkpoint')

            encoder = S2TPretrainedEncoder.get_class(cfg).build(cfg)
            encoder.load_state_dict(state['model'])
        else:
            pass

        return encoder

    @classmethod
    def build_decoder(cls, cfg: S2TPretrainedConfig, tgt_dict: Dictionary):
        if safe_getattr(cfg.decoder, "args") is None:
            state = load_checkpoint_to_cpu(cfg.decoder.path)
            if state.get("cfg", None) is not None:
                cfg.decoder.args = state['cfg']
            elif state.get("args", None) is not None:
                cfg.decoder.args = convert_namespace_to_omegaconf(state["args"])
            else:
                raise ValueError('Could not find args in checkpoint')

            decoder = S2TPretrainedDecoder.get_class(cfg).build(cfg, tgt_dict)
            decoder.load_state_dict(state['model'])
        else:
            pass

        return decoder


class S2TPretrainedEncoder(FairseqEncoder):
    """ Base class for pretrained S2T encoders """

    def __init__(self):
        FairseqEncoder.__init__(self, dictionary=None)

    @classmethod
    def get_class(cls, cfg: S2TPretrainedConfig):
        name = cfg.encoder.args.model._name
        if name.startswith('wav2vec'):
            return PretrainedWav2VecEncoder
        elif name.startswith('hubert'):
            return PretrainedHubertEncoder
        else:
            raise ValueError(f"Unknown encoder name: {name}")

    @classmethod
    def pre_build(cls, cfg: S2TPretrainedConfig) -> None:
        pass

    @classmethod
    def build(cls, cfg: S2TPretrainedConfig):
        cls.pre_build(cfg)
        return cls(cfg.encoder.args.model)

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

    def add_coupling_layers(self, cfg: FairseqDataclass):
        self.coupling_layers = nn.Linear(cfg.in_dim, cfg.out_dim)


class PretrainedWav2VecBaseEncoder(S2TPretrainedEncoder):
    """ Base class for pretrained wav2vec(-ish) encoders, including HuBERT """

    def __init__(self, cfg: FairseqDataclass):
        S2TPretrainedEncoder.__init__(self)

    @classmethod
    def pre_build(cls, cfg: S2TPretrainedEncoder):
        if cfg.encoder.args.model._name == cls.PRETRAIN_MODEL_NAME:
            model_args = DictConfig(
                cls.FINETUNE_MODEL_CFG(cls.FINETUNE_MODEL_NAME)
            )
        elif cfg.encoder.args.model._name == cls.FINETUNE_MODEL_NAME:
            model_args = merge_with_parent(
                cls.FINETUNE_MODEL_CFG,
                cfg.encoder.args.model,
                remove_missing=True
            )
        else:
            raise ValueError(
                f"Unknown model name: {cfg.encoder.args.model._name}"
            )

        if not safe_hasattr(model_args, 'w2v_args'):
            model_args.w2v_args = cfg.encoder.args
        
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

        cfg.encoder.args.model = model_args

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('w2v_model.'):
                if k.startswith('w2v_encoder.'):            # Finetuned models
                    k = re.sub(r'^w2v_encoder.', '', k)
                else:                                       # Pretrained models
                    k = f"w2v_model.{k}"
            if k.endswith('label_embs_concat') or k.startswith('proj.'):
                continue # This layers are not used
            new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict=strict)

    def pre_forward(self, src_tokens, src_lengths, **kwargs):
        encoder_in = S2TPretrainedEncoder.pre_forward(
            self, src_tokens, src_lengths, **kwargs
        )
        encoder_in['source'] = encoder_in.pop('src_tokens')
        encoder_in['padding_mask'] = lengths_to_padding_mask(src_lengths)
        return encoder_in

    def post_forward(self, encoder_out):
        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        encoder_out = {k: [v] for k, v in encoder_out.items()}
        S2TPretrainedEncoder.post_forward(self, encoder_out)

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

    def __init__(self, cfg: FairseqDataclass):
        PretrainedWav2VecBaseEncoder.__init__(self, cfg)
        Wav2VecEncoder.__init__(self, cfg)


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

    def __init__(self, cfg: FairseqDataclass):
        PretrainedWav2VecBaseEncoder.__init__(self, cfg)
        HubertEncoder.__init__(self, cfg, HubertDummyTask(cfg))


class S2TPretrainedDecoder(FairseqDecoder):
    """ Base class for pretrained S2T decoders """

    def __init__(self, tgt_dict: Dictionary):
        FairseqDecoder.__init__(self, dictionary=tgt_dict)

    @classmethod
    def get_class(cls, cfg: S2TPretrainedConfig):
        name = cfg.decoder.args.model._name
        if name.startswith('mbart'):
            return PretrainedBartDecoder
        else:
            raise ValueError(f"Unknown decoder name: {name}")

    @classmethod
    def pre_build(cls, cfg: S2TPretrainedConfig) -> None:
        pass

    @classmethod
    def build(cls, cfg: S2TPretrainedConfig, tgt_dict: Dictionary):
        cls.pre_build(cfg)
        return cls(cfg.decoder.args.model, tgt_dict)


class PretrainedBartDecoder(S2TPretrainedDecoder, TransformerDecoder):
    """ Pretrained BART decoder """

    def __init__(self, cfg: FairseqDataclass, tgt_dict: Dictionary, embed_tokens: nn.Embedding):
        S2TPretrainedDecoder.__init__(self, tgt_dict)
        TransformerDecoder.__init__(self, cfg, tgt_dict, embed_tokens)

    @classmethod
    def build(cls, cfg: S2TPretrainedConfig, tgt_dict: Dictionary):
        cls.pre_build(cfg)
        embed_tokens = TransformerModelBase.build_embedding(
            cfg.decoder.args.model,
            tgt_dict,
            cfg.decoder.args.model.decoder_embed_dim
        )
        return cls(cfg.decoder.args.model, tgt_dict, embed_tokens)

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

        super().load_state_dict(new_state_dict, strict=strict)
