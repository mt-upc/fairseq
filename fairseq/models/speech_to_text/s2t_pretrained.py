import re
import logging
from dataclasses import dataclass, field
from omegaconf import II, DictConfig, OmegaConf
from typing import Any, Optional, Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask, get_lengths
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
from fairseq.models.speechlm import SpeechLM, SpeechLMConfig
from fairseq.models.transformer import (
    TransformerModelBase,
    TransformerDecoder,
)
from fairseq.modules.adapter import ScaledParallelAdapter
from fairseq.modules.length_adaptor import (
    Conv1dAdaptorConfig,
    Conv1dAdaptor,
    ModalityAdapterConfig,
    ModalityAdapter,
)
from fairseq.modules import CTCDecoderConfig, CTCDecoder, TransformerEncoderLayers, EmbedderConfig, Embedder
from fairseq.models.transformer.transformer_config import TransformerConfig
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
class CouplingConfig(FairseqDataclass):
    conv1d_adaptor: Optional[Conv1dAdaptorConfig] = field(
        default=None,
        metadata={"help": "Conv1d Length Adaptor configuration"},
    )
    modality_adapter: Optional[ModalityAdapterConfig] = field(
        default=None,
        metadata={"help": "Modality Adapter configuration"},
    )
    context_encoder: Optional[TransformerConfig] = field(
        default=None,
        metadata={"help": "Context Encoder configuration"},
    )
    ctc_decoder: Optional[CTCDecoderConfig] = field(
        default=None,
        metadata={"help": "CTC Decoder configuration"},
    )
    embedder: Optional[EmbedderConfig] = field(
        default=None,
        metadata={"help": "Speech Embedder configuration"},
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
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "For how many updates to freeze the component in the start of training"}
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
    coupling: Optional[CouplingConfig] = field(
        default=None,
        metadata={"help": "coupling layers configuration"},
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
        if safe_hasattr(cfg, 'coupling'):
            encoder.add_coupling_modules(cfg.coupling)
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
            from fairseq.checkpoint_utils import load_checkpoint_to_cpu
            logger.info(f"Building {component_type} from: {cfg.path}")
            state = load_checkpoint_to_cpu(cfg.path)
            S2TPretrainedComponent.load_pre_args(cfg, state)

        if component_type == 'encoder':
            component = S2TPretrainedEncoder.get_class(cfg).build(cfg)
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
        self.frozen_params = []
        for n, p in self.named_parameters():
            for l in self.cfg_.layers_to_freeze:
                l = re.compile(eval(l))
                if re.match(l, n):
                    p.requires_grad = False
                    self.frozen_params.append(n)
        logger.info(
            f"Freezing parameters:\n\t" + '\n\t'.join(self.frozen_params)
        )

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates


class S2TPretrainedEncoder(FairseqEncoder, S2TPretrainedComponent):
    """ Base class for pretrained S2T encoders """

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        FairseqEncoder.__init__(self, dictionary=None)
        S2TPretrainedComponent.__init__(self, cfg)
        
        if safe_hasattr(cfg["pre_args"]["model"], "w2v_args"):
            self.embed_dim = cfg["pre_args"]["model"]["w2v_args"]["model"].encoder_embed_dim
        else:
            self.embed_dim = cfg["pre_args"]["model"]["encoder_embed_dim"]
        self.component_type = "ENCODER"
        self.coupling_modules = nn.ModuleList()

    @classmethod
    def get_class(cls, cfg: S2TPretrainedEncoderConfig) -> Type['S2TPretrainedEncoder']:
        name = cfg.pre_args.model._name
        if name.startswith('wav2vec'):
            return PretrainedWav2VecEncoder
        elif name.startswith('hubert'):
            return PretrainedHubertEncoder
        elif name.startswith('speechlm'):
            return PretrainedSpeechLMEncoder
        else:
            raise ValueError(f"Unknown encoder name: {name}")

    @classmethod
    def update_pre_args(cls, cfg: S2TPretrainedEncoderConfig) -> None:
        return S2TPretrainedComponent.update_pre_args(cfg)

    @classmethod
    def build(cls, cfg: S2TPretrainedEncoderConfig) -> 'S2TPretrainedEncoder':
        cls.update_pre_args(cfg)
        return cls(cfg)

    def add_coupling_modules(self, cfg: CouplingConfig) -> None:
        self.coupling_modules = nn.ModuleList()
        if cfg.ctc_decoder:
            if hasattr(cfg.ctc_decoder, "path") and cfg.ctc_decoder.path != "":
                logger.info(f"Loading CTCDecoder from: {cfg.ctc_decoder.path}")
                ckpt = torch.load(cfg.ctc_decoder.path, map_location='cpu')
                ckpt["cfg"].dropout_rate = cfg.ctc_decoder.dropout_rate
                
                ctc_dict = Dictionary.load(cfg.ctc_decoder.dictionary)
                # correct the ctc dictionary
                # TODO: make sure this is the same for HuBERT
                ctc_dict.symbols[0], ctc_dict.symbols[1] = ctc_dict.symbols[1], ctc_dict.symbols[0]
                ctc_dict.indices["<s>"], ctc_dict.indices["<pad>"] = 1, 0
                ctc_dict.bos_index, ctc_dict.pad_index = 1, 0

                ctc_decoder = CTCDecoder(ctc_dict, ckpt["cfg"])
                ctc_decoder.load_state_dict(ckpt["model"])
            else:
                # TODO initialize from pretrained speech encoder (if fine-tuned)
                ctc_decoder = CTCDecoder(cfg.ctc_decoder)
            # freeze the ctc decoder projection since there is no CTC loss
            # TODO: maybe also freeze post projection layer?
            ctc_decoder.proj.requires_grad = False
            logger.info(f"Freezing CTCDecoder projection layer")
            if hasattr(cfg.ctc_decoder, "freeze") and cfg.ctc_decoder.freeze:
                logger.info(f"Freezing CTCDecoder")
                for p in ctc_decoder.parameters():
                    p.requires_grad = False
            self.coupling_modules.append(ctc_decoder)
        if cfg.conv1d_adaptor:
            if hasattr(cfg.conv1d_adaptor, "path") and cfg.conv1d_adaptor.path != "":
                logger.info(f"Loading Conv1dAdaptor from: {cfg.conv1d_adaptor.path}")
                ckpt = torch.load(cfg.conv1d_adaptor.path, map_location='cpu')
                adaptor = Conv1dAdaptor(ckpt["cfg"])
                adaptor.load_state_dict(ckpt["model"])
            else:
                adaptor = Conv1dAdaptor(cfg.conv1d_adaptor)
            if hasattr(cfg.conv1d_adaptor, "freeze") and cfg.conv1d_adaptor.freeze:
                logger.info(f"Freezing Conv1dAdaptor")
                for p in adaptor.parameters():
                    p.requires_grad = False
            self.coupling_modules.append(adaptor)
        if cfg.modality_adapter:
            self.coupling_modules.append(
                ModalityAdapter(cfg.modality_adapter)
            )
        if cfg.embedder:
            if hasattr(cfg.embedder, "path") and cfg.embedder.path != "":
                logger.info(f"Loading Speech Embedder from: {cfg.embedder.path}")
                ckpt = torch.load(cfg.embedder.path, map_location='cpu')
                embedder = Embedder(ckpt["cfg"])
                embedder.load_state_dict(ckpt["model"])
            else:
                # TODO initialize from pretrained encoder-decoder
                embedder = Embedder(cfg.embedder)
            if hasattr(cfg.embedder, "freeze") and cfg.embedder.freeze:
                logger.info(f"Freezing Embedder")
                for p in embedder.parameters():
                    p.requires_grad = False
            self.coupling_modules.append(embedder)
        if cfg.context_encoder:
            if hasattr(cfg.context_encoder, "path") and cfg.context_encoder.path != "":
                logger.info(f"Loading ContextEncoder from: {cfg.context_encoder.path}")
                ckpt = torch.load(cfg.context_encoder.path, map_location='cpu')
                ckpt["cfg"].dropout = cfg.context_encoder.dropout
                ckpt["cfg"].attention_dropout = cfg.context_encoder.attention_dropout
                ckpt["cfg"].activation_dropout = cfg.context_encoder.activation_dropout
                context_encoder = TransformerEncoderLayers(ckpt["cfg"])
                context_encoder.load_state_dict(ckpt["model"])
            else:
                # TODO initialize from pretrained encoder-decoder
                context_encoder = TransformerEncoderLayers(cfg.context_encoder)
            self.coupling_modules.append(context_encoder)

    def pre_forward(self, src_tokens, src_lengths, **kwargs):
        return {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            **kwargs,
        }

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_inputs = self.pre_forward(src_tokens, src_lengths, **kwargs)
        encoder_out = self.ORIGINAL_MODEL_CLS.forward(self, **encoder_inputs)
        encoder_out["layer_results"] = None
        return self.post_forward(encoder_out)

    def post_forward(self, encoder_out):
        for i, (eo, epm) in enumerate(zip(encoder_out["encoder_out"], encoder_out["encoder_padding_mask"])):
            if safe_hasattr(self, 'coupling_modules'):
                for module in self.coupling_modules:
                    if isinstance(module, CTCDecoder):
                        speech_out = {
                            "encoder_out": [eo.transpose(0, 1)], # B x T x C
                            "encoder_out_lengths": [get_lengths(eo, epm)] # B
                        }
                        ctc_out = module(speech_out)  # T x B x V
                        _, speech_out = module.compress(ctc_out, speech_out)
                        if "modified_out" in speech_out:
                            eo = speech_out["modified_out"][0].transpose(0, 1) # T' x B x C
                            epm = speech_out["modified_padding_mask"][0] # B x T'
                    elif isinstance(module, Conv1dAdaptor) or isinstance(module, ModalityAdapter):
                        eo, epm = module(eo, epm) # T/n x B x C
                    elif isinstance(module, Embedder):
                        eo = eo.transpose(0, 1) # B x T x C
                        eo, epm, _ = module(eo, epm) # B x T+2 x C
                        eo = eo.transpose(0, 1) # T+2 x B x C
                    elif isinstance(module, TransformerEncoderLayers):
                        eo = module(eo, epm) # T x B x C
                    else:
                        raise NotImplementedError(f"Unknown coupling module: {module}")
            encoder_out["encoder_out"][i] = eo
            encoder_out["encoder_padding_mask"][i] = epm
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

        return super(S2TPretrainedEncoder, self).load_state_dict(new_state_dict, strict=strict)

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


class PretrainedSpeechLMEncoder(PretrainedWav2VecBaseEncoder):
    """ Pretrained SpeechLM encoder """

    PRETRAIN_MODEL_NAME = 'speechlm'
    FINETUNE_MODEL_NAME = 'speechlm_ctc'
    MODEL_CFG = SpeechLMConfig

    def __init__(self, cfg: S2TPretrainedEncoderConfig):
        PretrainedWav2VecBaseEncoder.__init__(self, cfg)
        speechlm_cfg_dict = OmegaConf.to_container(cfg.pre_args.model)
        self.speechlm = SpeechLM(SpeechLMConfig(speechlm_cfg_dict))

    @classmethod
    def build(cls, cfg: S2TPretrainedEncoderConfig) -> 'SpeechLM':
        return cls(cfg)

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_inputs = self.pre_forward(src_tokens, src_lengths, **kwargs)
        if self.speechlm.cfg.layer_norm_first:
            encoder_inputs['source'] = F.layer_norm(
                encoder_inputs['source'],
                encoder_inputs['source'][0].shape
            )  
        encoder_out = self.speechlm.forward(**encoder_inputs, features_only=True)
        encoder_out['encoder_out'] = encoder_out.pop('x').transpose(0, 1)
        _ = encoder_out.pop('features')
        _ = encoder_out.pop('layer_results')
        return self.post_forward(encoder_out)

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            k = f"speechlm.{k}"
            new_state_dict[k] = v

        return super(S2TPretrainedEncoder, self).load_state_dict(new_state_dict, strict=strict)


class S2TPretrainedDecoder(FairseqDecoder, S2TPretrainedComponent):
    """ Base class for pretrained S2T decoders """

    def __init__(self, cfg: S2TPretrainedDecoderConfig, tgt_dict: Dictionary):
        FairseqDecoder.__init__(self, dictionary=tgt_dict)
        S2TPretrainedComponent.__init__(self, cfg)
        
        self.embed_dim = cfg["pre_args"]["model"].decoder_embed_dim
        self.component_type = "DECODER"
        
    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        return super().forward(prev_output_tokens, encoder_out, **kwargs)

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
