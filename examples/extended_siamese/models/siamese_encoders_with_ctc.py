#!/usr/bin/env python3

import logging
from dataclasses import dataclass, field
from omegaconf import  OmegaConf, MISSING
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
)
from fairseq.models.transformer import Embedding, TransformerEncoder
from fairseq.models.wav2vec import Wav2VecEncoder
from examples.extended_siamese.modules import (
    CTCDecoder,
    CTCDecoderConfig,
    Adaptor,
    AdaptorConfig,
    ContextEncoder,
    ContextEncoderConfig,
    SpeechEmbedder,
    SpeechEmbedderConfig,
    Compressor,
    CompressorConfig
)


logger = logging.getLogger(__name__)

def _hasattr(obj, name):
    try:
        return hasattr(obj, name) and getattr(obj, name) is not None
    except:
        return False


class DummyDecoder(FairseqDecoder):
    def __init__(self, dictionary=None):
        super().__init__(dictionary)

@dataclass
class SpeechEncoderConfig:
    path: str = field(
        default=MISSING,
        metadata={"help": "path to pretrained speech encoder"}
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "speech encoder final dropout"}
    ) # TODO: this is not used. Remove later.
    dropout: float = field(
        default=0.0,
        metadata={"help": "speech encoder dropout"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "speech encoder attention dropout"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "speech encoder activation dropout"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "speech encoder dropout input"}
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "speech encoder layerdrop"}
    )
    apply_mask: bool = field(
        default=False,
        metadata={"help": "apply mask to speech encoder"}
    )
    mask_length: Optional[int] = field(
        default=None,
        metadata={"help": "speech encoder mask length"}
    )
    mask_channel_length: Optional[int] = field(
        default=None,
        metadata={"help": "speech encoder mask channel length"}
    )
    mask_prob: Optional[float] = field(
        default=None,
        metadata={"help": "speech encoder mask prob"}
    )
    mask_channel_prob: Optional[float] = field(
        default=None,
        metadata={"help": "speech encoder mask channel prob"}
    )
    ctc_layer_id: int = field(
        default=-1,
        metadata={"help": "in which layer to insert CTC module. -1 means the last layer."}
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "for how many updates to keep the speech encoder frozen"}
    )
    freeze_layers: int = field(
        default=-1,
        metadata={"help": "how many layers to freeze. -1 means None, 0 means feature extractor only, 1 means feature extractor + layer 0, etc."}
    )
    
@dataclass
class TextEncoderConfig(FairseqDataclass):
    path: str = field(
        default=MISSING,
        metadata={"help": "path to pretrained MT model"}
    )
    remove: bool = field(
        default=False,
        metadata={"help": "do not use text encoder."
                  "Remove it after loading the model to construct the speech embedder and context encoder"}
        )

@dataclass
class SiameseConfig(FairseqDataclass):
    speech_encoder: Optional[SpeechEncoderConfig] = None
    text_encoder: Optional[TextEncoderConfig] = None
    adaptor: Optional[AdaptorConfig] = None
    speech_embedder: Optional[SpeechEmbedderConfig] = None
    context_encoder: Optional[ContextEncoderConfig] = None
    ctc_decoder: Optional[CTCDecoderConfig] = None
    compressor: Optional[CompressorConfig] = None
    embed_dim: int = field(
        default=1024,
        metadata={"help": "model dimension"}
    )

class SiameseSpeechTextEncoders(FairseqEncoder):
    def __init__(
        self,
        cfg: SiameseConfig,
        spch_encoder,
        dictionary,
        text_encoder=None,
        adaptor=None,
        speech_embedder=None,
        context_encoder=None,
        compressor=None,
    ):
        super().__init__(dictionary)

        self.cfg = cfg
        self.spch_encoder = spch_encoder
        
        if text_encoder is not None:
            self.text_encoder = text_encoder
        if adaptor is not None:
            self.adaptor = adaptor
        if speech_embedder is not None:
            self.speech_embedder = speech_embedder
        if context_encoder is not None:
            self.context_encoder = context_encoder
        if compressor is not None:
            self.compressor = compressor

        self.ctc_layer_id = cfg.speech_encoder.ctc_layer_id
        
        self._freeze_text_encoder()  
        self._maybe_freeze_speech_encoder_layers()
        self._maybe_freeze_context_encoder()
        self._maybe_freeze_speech_embedder()
        
    @classmethod
    def build_speech_encoder(cls, cfg: SpeechEncoderConfig):
    
        try:
            ckpt = torch.load(cfg.path)
        except:
            ckpt = torch.load('/home/usuaris/scratch/ioannis.tsiamas/pretrained_models/wav2vec2/wav2vec_vox_960h_pl.pt')
        
        if "args" in ckpt and ckpt["args"] is not None:
            w2v_model_config = convert_namespace_to_omegaconf(ckpt["args"]).model
        else:
            w2v_model_config = OmegaConf.create(ckpt["cfg"]).model
        
        OmegaConf.set_struct(w2v_model_config, False)
        w2v_model_config.final_dropout = cfg.final_dropout
        w2v_model_config.w2v_args.model.dropout = cfg.dropout
        w2v_model_config.w2v_args.model.attention_dropout = cfg.attention_dropout
        w2v_model_config.w2v_args.model.activation_dropout = cfg.activation_dropout
        w2v_model_config.w2v_args.model.dropout_input = cfg.dropout_input
        w2v_model_config.apply_mask = cfg.apply_mask
        w2v_model_config.w2v_args.model.mask_length = cfg.mask_length
        w2v_model_config.w2v_args.model.mask_channel_length = cfg.mask_channel_length
        w2v_model_config.w2v_args.model.mask_prob = cfg.mask_prob
        w2v_model_config.w2v_args.model.mask_channel_prob = cfg.mask_channel_prob
        w2v_model_config.w2v_args.model.encoder_layerdrop = cfg.layerdrop
        w2v_model_config.ctc_layer_id = cfg.ctc_layer_id
        w2v_model_config.freeze_finetune_updates = cfg.freeze_finetune_updates
        OmegaConf.set_struct(w2v_model_config, True)
        
        spch_encoder = Wav2VecEncoder(
            w2v_model_config,
            output_size=ckpt["model"]["w2v_encoder.proj.weight"].size(0)
        )

        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("w2v_encoder."):
                model_ckpt[k.replace("w2v_encoder.", "")] = v
        
        logger.info(f"Loading Speech Model from {cfg.path} ...")
        missing_keys, unexpected_keys = spch_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        
        # these were not supposed to be here
        # so just putting zeros and freezing them
        if "hubert" in cfg.path:
            for l in range(7):
                spch_encoder.w2v_model.feature_extractor.conv_layers[l][0].bias.requires_grad = False
        
        spch_encoder.embed_dim = spch_encoder.w2v_model.cfg.encoder_embed_dim
        spch_encoder.w2v_model.encoder.ctc_layer_id = cfg.ctc_layer_id

        return spch_encoder

    @classmethod
    def build_text_encoder(cls, cfg: TextEncoderConfig, src_dictionary):
        
        try:
            ckpt = torch.load(cfg.path)
        except:
            ckpt = torch.load('/home/usuaris/scratch/ioannis.tsiamas/pretrained_models/nllb/nllb-200-distilled-600M.pt')
        
        if ckpt["args"] is None:
            model_args = ckpt["cfg"]["model"]
        else:
            model_args = ckpt["args"]
        
        enc_emb = Embedding(
            len(src_dictionary), model_args.encoder_embed_dim, src_dictionary.pad()
        )
        model_args.dropout = 0.0
        model_args.attention_dropout = 0.0
        model_args.activation_dropout = 0.0
        model_args.encoder_layerdrop = 0.0
        model_args.decoder_layerdrop = 0.0
        
        text_encoder = TransformerEncoder(
            model_args, src_dictionary, enc_emb, return_ln=True
        )
        
        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("encoder."):
                model_ckpt[k.replace("encoder.", "")] = v
        
        logger.info(f"Loading Text model from {cfg.path} ...")
        missing_keys, unexpected_keys = text_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))

        return text_encoder
    
    @classmethod
    def build_context_encoder(cls, cfg: ContextEncoderConfig, text_encoder):

        cfg.encoder.embed_dim = text_encoder.cfg.encoder.embed_dim
        cfg.encoder.ffn_embed_dim = text_encoder.cfg.encoder.ffn_embed_dim
        cfg.activation_fn = text_encoder.cfg.activation_fn
        cfg.encoder.normalize_before = text_encoder.cfg.encoder.normalize_before
        cfg.encoder.attention_heads = text_encoder.cfg.encoder.attention_heads
        cfg.encoder.layers = text_encoder.cfg.encoder.layers
        
        context_encoder = ContextEncoder(cfg, return_ln=True)

        context_encoder.layers.load_state_dict(text_encoder.layers.state_dict())
        if _hasattr(text_encoder, "layer_norm"):
            context_encoder.layer_norm.load_state_dict(text_encoder.layer_norm.state_dict())
        
        return context_encoder
    
    @classmethod
    def build_adaptor(cls, cfg: AdaptorConfig):
        adaptor = Adaptor(cfg)     
        return adaptor
    
    @classmethod
    def build_embedder(cls, cfg: SpeechEmbedderConfig, text_encoder):
        eos_token = "</s>"
        bos_token = "<lang:eng_Latn>"

        if not cfg.use_special_embedding and not cfg.use_positional_embedding:
            return None

        speech_embedder = SpeechEmbedder(cfg)
        
        if cfg.use_special_embedding and text_encoder is not None:
            
            def init_from_text(token):
                idx = text_encoder.dictionary.symbols.index(token)
                weights = text_encoder.embed_tokens.weight[idx].data
                return nn.Parameter(weights)
            
            speech_embedder.bos_emb = init_from_text(bos_token)
            speech_embedder.eos_emb = init_from_text(eos_token)
            logger.info("Loaded special embeddings for BOS and EOS from text encoder")
            
        if cfg.use_positional_embedding and cfg.learned_positional_embedding:
            
            speech_embedder.pos_emb.load_state_dict(text_encoder.embed_positions.state_dict())
            speech_embedder.layernorm.load_state_dict(text_encoder.layernorm_embedding.state_dict())
            logger.info("Loaded positional embedding and layernorm embedding from text encoder")
    
        return speech_embedder
    
    @classmethod
    def build_compressor(cls, cfg: CompressorConfig):
        compressor = Compressor(cfg)
        return compressor        
    
    def forward(self, src_tokens, src_lengths, src_txt_tokens, src_txt_lengths):
        raise NotImplementedError("Please use the forward submethods from the main model")
    
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
        
    def forward_adaptor(self, speech_out):
        
        if not _hasattr(self, "adaptor"):
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

    def forward_embedder(self, speech_out):
        if not _hasattr(self, "speech_embedder"):
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
        
        x, padding_mask, lengths = self.speech_embedder(x, padding_mask)
        
        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        if padding_mask is not None:
            assert padding_mask.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["modified_out"] = [x]
        speech_out["modified_out_lengths"] = [lengths]
        speech_out["modified_padding_mask"] = [padding_mask]
        
        return speech_out
    
    def forward_context(self, speech_out):
        if not _hasattr(self, "context_encoder"):
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

        if self.cfg.context_encoder.freeze:
            if self.context_encoder.training:
                self.context_encoder.eval()

        x = self.context_encoder(x, padding_mask)
            
        if isinstance(x, tuple):
            x, ln_results = x
        x = x.transpose(0, 1)

        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        if padding_mask is not None:
            assert padding_mask.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["context_out"] = [x]
        speech_out["context_out_lengths"] = speech_out[f"{key}_out_lengths"]
        speech_out["context_padding_mask"] = speech_out[f"{key}_padding_mask"]
        speech_out["context_ln_results"] = ln_results
        
        return speech_out
    
    def forward_compressor(self, decoder_out, speech_out):
        
        if not _hasattr(self, "compressor"):
            return speech_out
        
        with torch.no_grad():
            lprobs_ctc = F.log_softmax(decoder_out[0], dim=-1)  # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1)  # T x B
            preds = preds.transpose(0, 1)  # B x T
            
        x = speech_out["encoder_out"][0]  # B x T x C
        lens = speech_out["encoder_out_lengths"][0]  # B
        
        prev_lens = lens
        x, mask, lens, preds = self.compressor.char_compression(x, preds)
        
        speech_out["char_compression_rate"] = prev_lens.float() / lens.float()
        speech_out["compression_rate"] = speech_out["char_compression_rate"]

        if self.compressor.cfg.token_compression is not None:
            prev_lens = lens
            x, mask, lens = self.compressor.token_compression(x, prev_lens, preds)
            
            speech_out["token_compression_rate"] = prev_lens.float() / lens.float()
            speech_out["compression_rate"] = speech_out["token_compression_rate"] * speech_out["char_compression_rate"]
            
        speech_out["modified_out"] = [x]
        speech_out["modified_out_lengths"] = [lens]
        speech_out["modified_padding_mask"] = [mask]
        
        return speech_out
    
    def forward_text(self, src_txt_tokens, src_txt_lengths):
        if not _hasattr(self, "text_encoder"):
            return None
        if self.text_encoder.training:
            self.text_encoder.eval()
        return self.text_encoder(src_txt_tokens, src_txt_lengths, return_all_hiddens=True)
    
    def _freeze_text_encoder(self):
        if _hasattr(self, "text_encoder"):
            logger.info(f"Freezing text encoder ...")
            for n, p in self.text_encoder.named_parameters():
                logger.info(f"- freezing {n}")
                p.requires_grad = False
                
    def _maybe_freeze_speech_encoder_layers(self):
        if self.cfg.speech_encoder.freeze_layers > -1:
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
        if self.cfg.speech_encoder.freeze_layers > 0:
            logger.info(f"Freezing speech encoder layers ...")
            for i, layer in enumerate(self.spch_encoder.w2v_model.encoder.layers):
                if i < self.cfg.speech_encoder.freeze_layers:
                    for n, p in layer.named_parameters():
                        logger.info(f"- freezing layer{i} {n}")
                        p.requires_grad = False
                    layer.self_attn.dropout_module.p = 0.0
                    layer.dropout1.p = 0.0
                    layer.dropout2.p = 0.0
                    layer.dropout3.p = 0.0
                    
    def _maybe_freeze_context_encoder(self):
        if _hasattr(self, "context_encoder") and self.cfg.context_encoder.freeze:
            logger.info(f"Freezing context encoder ...")
            for n, p in self.context_encoder.named_parameters():
                logger.info(f"- freezing {n}")
                p.requires_grad = False
            logger.info(f"Not freezing self_attn_layer_norm of first layer ...")
            self.context_encoder.layers[0].self_attn_layer_norm.requires_grad = True
        # no need to deactivate dropout, it;s only gonna used during inference
        
    def _maybe_freeze_speech_embedder(self):
        if _hasattr(self, "speech_embedder") and self.cfg.speech_embedder.freeze:
            logger.info("Freezing speech embedder ...")
            for n, p in self.speech_embedder.named_parameters():
                logger.info(f"- freezing {n}")
                p.requires_grad = False

@register_model("siamese_encoders_with_ctc", dataclass=SiameseConfig)
class SiameseEncodersWithCTC(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0
        
    @classmethod
    def build_encoder(cls, cfg: SiameseConfig, src_dict):
        spch_encoder = SiameseSpeechTextEncoders.build_speech_encoder(cfg.speech_encoder)
        text_encoder = SiameseSpeechTextEncoders.build_text_encoder(cfg.text_encoder, src_dict) if _hasattr(cfg, "text_encoder") else None
        adaptor = SiameseSpeechTextEncoders.build_adaptor(cfg.adaptor) if _hasattr(cfg, "adaptor") else None
        context_encoder = SiameseSpeechTextEncoders.build_context_encoder(cfg.context_encoder, text_encoder) if _hasattr(cfg, "context_encoder") else None
        speech_embedder = SiameseSpeechTextEncoders.build_embedder(cfg.speech_embedder, text_encoder) if _hasattr(cfg, "speech_embedder") else None
        compressor = SiameseSpeechTextEncoders.build_compressor(cfg.compressor) if _hasattr(cfg, "compressor") else None
        
        encoder = SiameseSpeechTextEncoders(
            cfg,
            spch_encoder,
            src_dict,
            text_encoder=text_encoder if not cfg.text_encoder.remove else None,
            adaptor=adaptor,
            context_encoder=context_encoder,
            speech_embedder=speech_embedder,
            compressor=compressor
        )
        
        return encoder

    @classmethod
    def build_decoder(cls, cfg: CTCDecoderConfig, encoder):
        if cfg is not None:
            decoder = CTCDecoder(cfg)
            decoder.proj = encoder.spch_encoder.proj
        else:
            decoder = DummyDecoder()
        return decoder

    @classmethod
    def build_model(cls, cfg, task):       
        encoder = cls.build_encoder(cfg, task.src_dict)
        decoder = cls.build_decoder(cfg.ctc_decoder, encoder)
        
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

        if _hasattr(self, "adaptive_softmax"):
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
            decoder_out = self.decoder(speech_out)

            speech_out = self.encoder.forward_compressor(decoder_out, speech_out)

        speech_out = self.encoder.forward_adaptor(speech_out)
        
        speech_out = self.encoder.forward_embedder(speech_out)

        speech_out = self.encoder.forward_context(speech_out)
        
        text_out = self.encoder.forward_text(src_txt_tokens, src_txt_lengths)
        
        return decoder_out, (speech_out, text_out)