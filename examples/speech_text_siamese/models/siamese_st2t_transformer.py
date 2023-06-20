#!/usr/bin/env python3

import logging

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerEncoder, TransformerDecoder
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.modules import (
    CTCDecoder,
    CTCDecoderConfig,
    Adaptor,
    AdaptorConfig,
    TransformerEncoderLayers,
    Embedder,
    EmbedderConfig
)
from fairseq.models.transformer.transformer_config import TransformerConfig

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class DummyDecoder(FairseqDecoder):
    def __init__(self, dictionary=None):
        super().__init__(dictionary)


class SiameseSpeechTextEncoders(FairseqEncoder):
    def __init__(
        self,
        args,
        spch_encoder,
        dictionary,
        text_encoder=None,
        adaptor=None,
        embedder=None,
        context_encoder=None,
    ):
        super().__init__(dictionary)

        self.args = args
        self.spch_encoder = spch_encoder
        
        if text_encoder is not None:
            self.text_encoder = text_encoder
        if adaptor is not None:
            self.adaptor = adaptor
        if embedder is not None:
            self.embedder = embedder
        if context_encoder is not None:
            self.context_encoder = context_encoder

        self.ctc_layer_id = args.w2v_ctc_layer_id
        self.retain_dropout_in_frozen_text_encoder = args.retain_dropout_in_frozen_text_encoder
        self.freeze_speech_encoder_layers = args.freeze_speech_encoder_layers

        self._freeze_text_encoder()  
        self._maybe_freeze_speech_encoder_layers()
        self._maybe_freeze_context_encoder()

    @classmethod
    def build_speech_encoder(cls, args):
    
        ckpt = torch.load(args.speech_model_path)
        
        w2v_args = ckpt["args"]
        w2v_model_config = convert_namespace_to_omegaconf(w2v_args).model
        
        OmegaConf.set_struct(w2v_model_config, False)
        w2v_model_config.final_dropout = args.w2v_final_dropout
        w2v_model_config.w2v_args.model.dropout = args.w2v_dropout
        w2v_model_config.w2v_args.model.attention_dropout = args.w2v_attention_dropout
        w2v_model_config.w2v_args.model.activation_dropout = args.w2v_activation_dropout
        w2v_model_config.w2v_args.model.dropout_input = args.w2v_dropout_input
        w2v_model_config.apply_mask = args.w2v_apply_mask
        w2v_model_config.w2v_args.model.mask_length = args.w2v_mask_length
        w2v_model_config.w2v_args.model.mask_channel_length = args.w2v_mask_channel_length
        w2v_model_config.w2v_args.model.mask_prob = args.w2v_mask_prob
        w2v_model_config.w2v_args.model.mask_channel_prob = args.w2v_mask_channel_prob
        w2v_model_config.freeze_finetune_updates = 0
        w2v_model_config.w2v_args.model.encoder_layerdrop = args.w2v_layerdrop
        w2v_model_config.ctc_layer_id = args.w2v_ctc_layer_id
        OmegaConf.set_struct(w2v_model_config, True)
        
        spch_encoder = Wav2VecEncoder(
            w2v_model_config,
            output_size=ckpt["model"]["w2v_encoder.proj.weight"].size(0)
        )

        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("w2v_encoder."):
                model_ckpt[k.replace("w2v_encoder.", "")] = v
        
        logger.info(f"Loading Speech Model from {args.speech_model_path} ...")
        missing_keys, unexpected_keys = spch_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        
        spch_encoder.embed_dim = spch_encoder.w2v_model.cfg.encoder_embed_dim
        spch_encoder.w2v_model.encoder.ctc_layer_id = args.w2v_ctc_layer_id
        
        args.w2v_cfg = w2v_model_config

        return spch_encoder

    @classmethod
    def build_text_encoder(cls, args, src_dictionary):
        if args.only_ctc:
            return None
        
        ckpt = torch.load(args.text_encoder_path)
        
        if ckpt["args"] is None:
            model_args = ckpt["cfg"]["model"]
        else:
            model_args = ckpt["args"]
        
        enc_emb = Embedding(
            len(src_dictionary), model_args.encoder_embed_dim, src_dictionary.pad()
        )
        
        if not args.retain_dropout_in_frozen_text_encoder:
            model_args.dropout = 0.0
            model_args.attention_dropout = 0.0
            model_args.activation_dropout = 0.0
            model_args.encoder_layerdrop = 0.0
            model_args.decoder_layerdrop = 0.0
        
        text_encoder = TransformerEncoder(
            model_args, src_dictionary, enc_emb
        )
        
        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("encoder."):
                model_ckpt[k.replace("encoder.", "")] = v
        
        logger.info(f"Loading Text model from {args.text_encoder_path} ...")
        missing_keys, unexpected_keys = text_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))

        return text_encoder
    
    @classmethod
    def build_context_encoder(cls, args, text_encoder):
        if args.only_ctc or not args.use_context_encoder:
            return None
        
        transformer_cfg = TransformerConfig()
        transformer_cfg.encoder.embed_dim = text_encoder.cfg.encoder.embed_dim
        transformer_cfg.encoder.ffn_embed_dim = text_encoder.cfg.encoder.ffn_embed_dim
        transformer_cfg.activation_fn = text_encoder.cfg.activation_fn
        transformer_cfg.encoder.normalize_before = text_encoder.cfg.encoder.normalize_before
        transformer_cfg.encoder.attention_heads = text_encoder.cfg.encoder.attention_heads
        transformer_cfg.encoder.layers = text_encoder.cfg.encoder.layers
        
        transformer_cfg.dropout = args.context_dropout
        transformer_cfg.activation_dropout = args.context_activation_dropout
        transformer_cfg.attention_dropout = args.context_attention_dropout
        
        context_encoder = TransformerEncoderLayers(transformer_cfg)

        context_encoder.layers.load_state_dict(text_encoder.layers.state_dict())
        if hasattr(text_encoder, "layer_norm"):
            context_encoder.layer_norm.load_state_dict(text_encoder.layer_norm.state_dict())
            
        args.context_encoder_cfg = transformer_cfg
        
        return context_encoder
    
    @classmethod
    def build_adaptor(cls, args, spch_encoder):
        if args.only_ctc:
            return None
        
        if not args.adaptor_layers \
        and not args.adaptor_pre_projection \
        and not args.adaptor_post_projection:
            return None
        
        adaptor_cfg = AdaptorConfig()
        adaptor_cfg.embed_dim = spch_encoder.embed_dim
        adaptor_cfg.num_layers = args.adaptor_layers
        adaptor_cfg.kernel_size = args.adaptor_kernel_size
        adaptor_cfg.stride = args.adaptor_stride
        adaptor_cfg.pre_projection = args.adaptor_pre_projection
        adaptor_cfg.post_projection = args.adaptor_post_projection
        adaptor_cfg.projection_dim = args.adaptor_projection_dim
        adaptor_cfg.dropout_rate = args.adaptor_dropout
        adaptor_cfg.use_final_layer_norm = args.adaptor_final_layer_norm
        # adaptor_cfg.use_final_layer_norm = False
        
        adaptor = Adaptor(adaptor_cfg)
        
        args.adaptor_args = adaptor_cfg
            
        return adaptor
    
    @classmethod
    def build_embedder(cls, args, spch_encoder, text_encoder):
        eos_token = "</s>"
        
        if "<lang:eng_Latn>" in text_encoder.dictionary.symbols:
            model_name="nllb"
        else:
            model_name="mbart"

        if model_name == "nllb":
            bos_token = "<lang:eng_Latn>"
            learned_pos = False
            max_positions = 512
        else:
            bos_token = "<lang:en>"
            learned_pos = True
            max_positions = 1024
        
        if args.only_ctc:
            return None
        if not args.use_special_embedding and not args.use_positional_embedding:
            return None
        
        embedder_cfg = EmbedderConfig()
        embedder_cfg.use_special_embedding = args.use_special_embedding
        embedder_cfg.use_positional_embedding = args.use_positional_embedding
        embedder_cfg.scale_embedding = args.scale_embedding
        embedder_cfg.embed_dim = spch_encoder.embed_dim
        embedder_cfg.max_source_positions = max_positions
        embedder_cfg.is_learned = learned_pos
        embedder_cfg.layer_norm_special = args.adaptor_final_layer_norm
        embedder = Embedder(embedder_cfg)
        
        if text_encoder is not None:
            if args.use_special_embedding:
                bos_idx = text_encoder.dictionary.symbols.index(bos_token)
                weights_bos = text_encoder.embed_tokens.weight[bos_idx].data
                embedder.bos_emb = nn.Parameter(weights_bos)
                
                eos_idx = text_encoder.dictionary.symbols.index(eos_token)
                weights_eos = text_encoder.embed_tokens.weight[eos_idx].data
                embedder.eos_emb = nn.Parameter(weights_eos)
                logger.info("Loaded special embeddings for BOS and EOS from text encoder")
            if args.use_positional_embedding and learned_pos:
                embedder.pos_emb.load_state_dict(text_encoder.embed_positions.state_dict())
                embedder.layernorm.load_state_dict(text_encoder.layernorm_embedding.state_dict())
                logger.info("Loaded positional embedding and layernorm embedding from text encoder")
        
        args.embedder_args = embedder_cfg
        
        return embedder
    
    def forward(self, src_tokens, src_lengths, src_txt_tokens, src_txt_lengths):
        raise NotImplementedError("Please use the forward submethods from the main model")
    
    def forward_adaptor(self, speech_out):
        
        if not hasattr(self, "adaptor"):
            return speech_out
        
        if "modified_out" in speech_out:
            key = "modified"
        else:
            key = "encoder"
        
        x = speech_out[f"{key}_out"][0].transpose(0, 1)
        padding_mask = speech_out[f"{key}_padding_mask"][0]
        
        x, padding_mask = self.adaptor(x, padding_mask)
        x = x.transpose(0, 1)
        
        if padding_mask is not None:
            output_lengths = (1 - padding_mask.int()).sum(dim=1)
        else:
            B, T, _ = x.size()
            output_lengths = (torch.ones(B, device=x.device) * T).long()
        
        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        if padding_mask is not None:
            assert padding_mask.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["modified_out"] = [x]
        speech_out["modified_padding_mask"] = [padding_mask]
        speech_out["modified_out_lengths"] = [output_lengths]
        
        return speech_out
    
    def forward_speech(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        
        w2v_args = {
            "source": src_tokens,
            "padding_mask": padding_mask,
            "mask": self.spch_encoder.apply_mask and self.spch_encoder.training,
        }
        
        res = self.spch_encoder.w2v_model.extract_features(**w2v_args)

        padding_mask = res["padding_mask"]
        if padding_mask is not None:
            output_lengths = (1 - padding_mask.int()).sum(dim=1)
        else:
            B, T, _ = res["x"].size()
            output_lengths = (torch.ones(B, device=res["x"].device) * T).long()
            
        return {
            "encoder_out": [res["x"]],  # B x T x C
            "encoder_padding_mask": [padding_mask], # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "encoder_out_lengths": [output_lengths],
            "ctc_layer_result": res["layer_results"][self.ctc_layer_id] if self.ctc_layer_id != -1 else None,
        }
        
    def forward_context(self, speech_out):
        if not hasattr(self, "context_encoder"):
            return speech_out
        
        if "modified_out" in speech_out:
            key = "modified"
        else:
            key = "encoder"
        
        x = speech_out[f"{key}_out"][0].transpose(0, 1)
        if speech_out[f"{key}_padding_mask"][0] is not None:
            padding_mask = speech_out[f"{key}_padding_mask"][0]
        else:
            padding_mask = None

        x = self.context_encoder(x, padding_mask)
        x = x.transpose(0, 1)

        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        if padding_mask is not None:
            assert padding_mask.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["context_out"] = [x]
        speech_out["context_out_lengths"] = speech_out[f"{key}_out_lengths"]
        speech_out["context_padding_mask"] = speech_out[f"{key}_padding_mask"]
        
        return speech_out
    
    def forward_embedder(self, speech_out):
        if not hasattr(self, "embedder"):
            return speech_out
        
        if "modified_out" in speech_out:
            key = "modified"
        else:
            key = "encoder"
        
        x = speech_out[f"{key}_out"][0]
        lengths = speech_out[f"{key}_out_lengths"][0]
        if speech_out[f"{key}_padding_mask"][0] is not None:
            padding_mask = speech_out[f"{key}_padding_mask"][0]
        else:
            padding_mask = None
        assert x.size(0) == lengths.size(0)
        
        x, padding_mask, lengths = self.embedder(x, padding_mask)
        
        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        if padding_mask is not None:
            assert padding_mask.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["modified_out"] = [x]
        speech_out["modified_out_lengths"] = [lengths]
        speech_out["modified_padding_mask"] = [padding_mask]
        
        return speech_out
    
    def forward_text(self, src_txt_tokens, src_txt_lengths):
        if not hasattr(self, "text_encoder"):
            return None
        if self.text_encoder.training:
            self.text_encoder.eval()
        return self.text_encoder(src_txt_tokens, src_txt_lengths)
    
    def _freeze_text_encoder(self):
        if hasattr(self, "text_encoder"):
            logger.info(f"Freezing text encoder ...")
            for n, p in self.text_encoder.named_parameters():
                logger.info(f"- freezing {n}")
                p.requires_grad = False
                
    def _maybe_freeze_speech_encoder_layers(self):
        if self.freeze_speech_encoder_layers > -1:
            logger.info(f"Freezing speech encoder feature extractor ...")
            ft_layers = [
                ("feature_extractor", self.spch_encoder.w2v_model.feature_extractor),
                ("post_extract_proj", self.spch_encoder.w2v_model.post_extract_proj),
                ("pos_conv", self.spch_encoder.w2v_model.encoder.pos_conv)
            ]
            for name, layer in ft_layers:
                for n, p in layer.named_parameters():
                    logger.info(f"- freezing {name} {n}")
                    p.requires_grad = False
        
        # TODO there are some more dropouts that need to be frozen *also masking
        if self.freeze_speech_encoder_layers > 0:
            logger.info(f"Freezing speech encoder layers ...")
            for i, layer in enumerate(self.spch_encoder.w2v_model.encoder.layers):
                if i < self.freeze_speech_encoder_layers:
                    for n, p in layer.named_parameters():
                        logger.info(f"- freezing layer{i} {n}")
                        p.requires_grad = False
                    layer.self_attn.dropout_module.p = 0.0
                    layer.dropout1.p = 0.0
                    layer.dropout2.p = 0.0
                    layer.dropout3.p = 0.0
                    
    def _maybe_freeze_context_encoder(self):
        if hasattr(self, "context_encoder") and not self.args.ot_weight:
            logger.info(f"Freezing context encoder ...")
            for n, p in self.context_encoder.named_parameters():
                logger.info(f"- freezing {n}")
                p.requires_grad = False
        # no need to deactivate dropout, it;s only gonna used during inference

@register_model("siamese_st2t_transformer")
class SiameseST2TTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--adaptor-layers", type=int, default=0, help="number of layers in adaptor"
        )
        parser.add_argument(
            "--adaptor-kernel-size", type=int, default=3, help="kernel size in adaptor"
        )
        parser.add_argument(
            "--adaptor-stride", type=int, default=2, help="stride in adaptor"
        )
        parser.add_argument(
            "--adaptor-pre-projection", type=str, default="false",
            help="apply pre-projection to adaptor"
        )
        parser.add_argument(
            "--adaptor-post-projection", type=str, default="false",
            help="apply post-projection to adaptor"
        )
        parser.add_argument(
            "--adaptor-dropout", type=float, default=0.0, help="dropout in adaptor"
        )
        parser.add_argument(
            "--adaptor-final-layer-norm", type=str, default="false",
            help="apply layer norm to adaptor output"
        )
        parser.add_argument(
            "--adaptor-projection-dim", type=int, default=8192,
            help="projection dimension in adaptor"
        )
        parser.add_argument(
            "--ctc-compression", type=str, default="false",
            help="Apply CTC compression to speech encoder's output",
        )
        parser.add_argument(
            "--speech-model-path", type=str, default="", help="path to pre-trained speech model"
        )
        parser.add_argument(
            "--severed-layer",
            type=int,
            default=None,
            help="speech encoder layer to insert CTC module",
        ) 
        parser.add_argument(
            "--text-encoder-path", type=str, default="", help="path to pre-trained text model for the encoder"
        )
        parser.add_argument(
            "--retain-dropout-in-frozen-text-encoder", type=str, default="false",
            help="retain dropout in frozen mbart text encoder"
        )
        parser.add_argument(
            "--w2v-final-dropout", type=float, metavar="D", help="wav2vec speech encoder final dropout"
        )
        parser.add_argument(
            "--w2v-dropout", type=float, metavar="D", help="wav2vec speech encoder dropout"
        )
        parser.add_argument(
            "--w2v-attention-dropout", type=float, metavar="D", help="wav2vec speech encoder attention dropout"
        )
        parser.add_argument(
            "--w2v-activation-dropout", type=float, metavar="D", help="wav2vec speech encoder activation dropout"
        )
        parser.add_argument(
            "--w2v-dropout-input", type=float, metavar="D", help="wav2vec speech encoder dropout input"
        )
        parser.add_argument(
            "--w2v-layerdrop", type=float, metavar="D", help="wav2vec speech encoder layerdrop"
        )
        parser.add_argument(
            "--w2v-apply-mask", type=str, default="false", help="apply mask to wav2vec speech encoder"
        )
        parser.add_argument(
            "--w2v-mask-length", type=int, metavar="D", help="wav2vec speech encoder mask length"
        )
        parser.add_argument(
            "--w2v-mask-channel-length", type=int, metavar="D", help="wav2vec speech encoder mask channel length"
        )
        parser.add_argument(
            "--w2v-mask-prob", type=float, metavar="D", help="wav2vec speech encoder mask prob"
        )
        parser.add_argument(
            "--w2v-mask-channel-prob", type=float, metavar="D", help="wav2vec speech encoder mask channel prob"
        )
        parser.add_argument(
            "--w2v-ctc-layer-id", type=int, default=-1, help="in which layer to insert CTC module"
        )
        parser.add_argument(
            "--use-context-encoder", type=str, default="false",
            help="use a text encoder on top of the speech encoder"
        )
        parser.add_argument(
            "--context-dropout", type=float, metavar="D", help="context encoder dropout"
        )
        parser.add_argument(
            "--context-attention-dropout", type=float, metavar="D", help="context encoder attention dropout"
        )
        parser.add_argument(
            "--context-activation-dropout", type=float, metavar="D", help="context encoder activation dropout"
        )
        parser.add_argument(
            "--freeze-speech-encoder-layers", type=int, default=-1,
            help="freeze the speech encoder layers with id < freeze-speech-encoder-layers"
            "-1 means no freezing, 0 means freeze feature-extractor, 1 means freeze feature-extractor and encoder.layer.0, etc..."
        )
        parser.add_argument(
            "--no-text-encoder", type=str, default="false",
            help="do not use text encoder (no OT), use only speech encoder"
        )
        parser.add_argument(
            "--use-special-embedding", type=str, default="false",
            help="use bos and eos embedding in the output of the speech encoder"
        )
        parser.add_argument(
            "--use-positional-embedding", type=str, default="false",
            help="use positional embedding in the output of the speech encoder"
        )
        parser.add_argument(
            "--scale-embedding", type=str, default="false",
            help="scale the representation before applying the positional embedding"
            "in the embedder module."
        )
        parser.add_argument(
            "--ctc-compression-type", type=str, default="letter", choices=["letter", "word"],
            help="type of CTC compression"
        )
        parser.add_argument(
            "--ctc-compression-pooling-fn", type=str, default="mean", choices=["mean", "max", "attention"],
            help="pooling function to use for CTC compression"
        )
        parser.add_argument(
            "--ctc-dictionary", type=str, default="",
            help="path to the dictionary for CTC compression"
        )
        
    @classmethod
    def build_encoder(cls, args, task):
        spch_encoder = SiameseSpeechTextEncoders.build_speech_encoder(args)
        text_encoder = SiameseSpeechTextEncoders.build_text_encoder(args, task.src_dict)
        adaptor = SiameseSpeechTextEncoders.build_adaptor(args, spch_encoder)
        context_encoder = SiameseSpeechTextEncoders.build_context_encoder(args, text_encoder)
        embedder = SiameseSpeechTextEncoders.build_embedder(args, spch_encoder, text_encoder)
        
        encoder = SiameseSpeechTextEncoders(
            args,
            spch_encoder,
            task.src_dict,
            text_encoder=text_encoder if not args.no_text_encoder else None,
            adaptor=adaptor,
            context_encoder=context_encoder,
            embedder=embedder
        )
        
        return encoder

    @classmethod
    def build_decoder(cls, args, task, encoder):
        
        if args.ctc_weight > 0.0:
            ctc_decoder_cfg = CTCDecoderConfig()
            ctc_decoder_cfg.embed_dim = encoder.spch_encoder.embed_dim
            ctc_decoder_cfg.dropout_rate = encoder.spch_encoder.final_dropout.p
            ctc_decoder_cfg.ctc_compression = args.ctc_compression
            ctc_decoder_cfg.ctc_compression_type = args.ctc_compression_type
            ctc_decoder_cfg.pooling_fn = args.ctc_compression_pooling_fn
            ctc_decoder_cfg.layernorm = args.w2v_ctc_layer_id != -1
            ctc_decoder_cfg.final_layernorm = True
            ctc_decoder_cfg.dictionary_path = args.ctc_dictionary
                
            decoder = CTCDecoder(ctc_decoder_cfg)
            # initialized the decoder's projection layer with the encoder's ctc projection layer
            decoder.proj = encoder.spch_encoder.proj
            
            args.ctc_decoder_cfg = ctc_decoder_cfg
            
        else:
            decoder = DummyDecoder()
            
        return decoder

    @classmethod
    def build_model(cls, args, task, process_args=True):
        
        if process_args:
            default_bool = lambda x: getattr(args, x, "false") == "true"
            
            args.w2v_ctc_layer_id = getattr(args, "w2v_ctc_layer_id", -1)
            args.adaptor_layers = getattr(args, "adaptor_layers", 0)
            args.adaptor_projection_dim = getattr(args, "adaptor_projection_dim", 8192)
            args.freeze_speech_encoder_layers = getattr(args, "freeze_speech_encoder_layers", -1)
            args.ctc_compression_type = getattr(args, "ctc_compression_type", "letter")
            args.ctc_compression_pooling_fn = getattr(args, "ctc_compression_pooling_fn", "mean")
            args.adaptor_dropout = getattr(args, "adaptor_dropout", 0.0)
            args.use_context_encoder = default_bool("use_context_encoder")
            args.retain_dropout_in_frozen_text_encoder = default_bool("retain_dropout_in_frozen_text_encoder")
            args.ctc_compression = default_bool("ctc_compression")
            args.w2v_apply_mask = default_bool("w2v_apply_mask")
            args.no_text_encoder = default_bool("no_text_encoder")
            args.use_special_embedding = default_bool("use_special_embedding")
            args.use_positional_embedding = default_bool("use_positional_embedding")
            args.scale_embedding = default_bool("scale_embedding")
            args.adaptor_pre_projection = default_bool("adaptor_pre_projection")
            args.adaptor_post_projection = default_bool("adaptor_post_projection")
            args.adaptor_final_layer_norm = default_bool("adaptor_final_layer_norm")
            args.only_ctc = not args.ot_weight and not args.ot_emb_weight
        
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task, encoder)
        
        # do it after initializing the decoder to transfer the ctc weights
        encoder.spch_encoder.w2v_model.remove_pretraining_modules()
        encoder.spch_encoder.proj = None
        
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None, idx=0):
        lprobs = self.get_normalized_probs_scriptable(
            net_output, log_probs, sample, idx=idx
        )
        lprobs.batch_first = False
        return lprobs

    def get_normalized_probs_scriptable(self, net_output, log_probs, sample, idx=0):
        """Get normalized probabilities (or log probs) from a net's output."""
        assert not isinstance(self.decoder, DummyDecoder)

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def forward_torchscript(self, net_input):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        encoder_out = self.forward(**encoder_input)[1][0]
        
        if "context_out" in encoder_out:
            key = "context"
        elif "modified_out" in encoder_out:
            key = "modified"
        else:
            key = "encoder"

        encoder_out = {
            "encoder_out": [encoder_out[f"{key}_out"][0].transpose(0, 1)],
            "encoder_padding_mask": encoder_out[f"{key}_padding_mask"]
        }
        return encoder_out

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
        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
        }

    def forward(self, src_tokens, src_lengths, src_txt_tokens=None, src_txt_lengths=None):
        speech_out = self.encoder.forward_speech(src_tokens, src_lengths)

        decoder_out = None
        if isinstance(self.decoder, CTCDecoder):
            breakpoint()
            decoder_out = self.decoder(speech_out)

            if self.decoder.cfg.ctc_compression:
                decoder_out, speech_out = self.decoder.compress(decoder_out, speech_out)

        speech_out = self.encoder.forward_adaptor(speech_out)
        
        speech_out = self.encoder.forward_embedder(speech_out)

        if self.encoder.args.ot_weight > 0.0:
            speech_out = self.encoder.forward_context(speech_out)
        
        text_out = self.encoder.forward_text(src_txt_tokens, src_txt_lengths)
        
        return decoder_out, (speech_out, text_out)
    

@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer")
def siamese_st2t_transformer_base(args):
    pass

@register_model("siamese_zs_transformer")
class SiameseZSTransformer(BaseFairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--encoder-path", type=str, required=True,
            help="path to encoder model"
        )
        parser.add_argument(
            "--decoder-path", type=str, required=True,
            help="path to decoder model"
        )
        parser.add_argument(
            "--not-load-submodules", type=str, default="false",
        )

    @staticmethod
    def build_encoder(args, task):
        ckpt = torch.load(args.encoder_path)
        model_args = ckpt["cfg"]["model"]
        model_args.no_text_encoder = True
        task.src_dict = task.tgt_dict
        encoder = SiameseST2TTransformerModel.build_model(model_args, task, process_args=False)
        if not args.not_load_submodules:
            logger.info(f"Loading encoder from {args.encoder_path} ...")
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt["model"], strict=False)
            if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
            if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        return encoder
    
    @staticmethod
    def build_decoder(args, task):
       
        ckpt = torch.load(args.decoder_path)
        model_args = ckpt["cfg"]["model"]
        
        dec_emb = Embedding(
            len(task.tgt_dict), model_args.decoder_embed_dim, task.tgt_dict.pad()
        )
        decoder = TransformerDecoder(model_args, task.tgt_dict, dec_emb)
        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("decoder."):
                model_ckpt[k.replace("decoder.", "")] = v
        
        if not args.not_load_submodules:
            logger.info(f"Loading decoder from {args.decoder_path} ...")
            missing_keys, unexpected_keys = decoder.load_state_dict(model_ckpt, strict=False)
            if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
            if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        
        return decoder
    
    @classmethod
    def build_model(cls, args, task):
        default_bool = lambda x: getattr(args, x, "false") == "true"
        args.not_load_submodules = default_bool("not_load_submodules")
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder_out = {
            "encoder_out": [encoder_out[1][0]["encoder_out"][0].transpose(0, 1)],
            "encoder_padding_mask": encoder_out[1][0]["encoder_padding_mask"]
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def max_positions(self):
        return (self.encoder.max_positions()[0], self.decoder.max_positions())

    def max_decoder_positions(self):
        return self.decoder.max_positions()
    
@register_model_architecture("siamese_zs_transformer", "siamese_zs_transformer")
def siamese_zs_transformer_base(args):
    pass