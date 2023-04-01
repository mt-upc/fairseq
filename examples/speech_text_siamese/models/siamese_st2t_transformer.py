#!/usr/bin/env python3

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerEncoder
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.modules import FairseqDropout, LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.length_adaptor import Conv1dAdaptor, Conv1dAdaptorConfig
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class DummyDecoder(FairseqDecoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    logging.info(f"| bias in Linear layer: {bias}")
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    return Embedding(num_embeddings, embed_dim, padding_idx)


class TransformerEncoderLayers(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = nn.ModuleList(
            [TransformerEncoderLayerBase(cfg) for _ in range(cfg.encoder.layers)]
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.layer_norm = LayerNorm(cfg.encoder.embed_dim)

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        
        if not self.normalize_before:
            x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        if self.normalize_before:
            x = self.layer_norm(x)
        return x
    

class CTCDecoder(FairseqDecoder):
    def __init__(self, dictionary, embed_dim, dropout_rate=0.0, ctc_compression=False, layernorm=False):
        super().__init__(dictionary)
        
        self.dictionary = dictionary
        self.embed_dim = embed_dim        
        self.blank_idx = dictionary.pad() # TODO have to check for other encoders.
        
        # only if the expected input is not the final output of the speech encoder
        if layernorm:
            self.layer_norm = LayerNorm(embed_dim)
            
        self.dropout_module = FairseqDropout(dropout_rate)
        self.proj = Linear(embed_dim, len(dictionary), bias=True)
        
        self.ctc_compression = ctc_compression
        self.min_frames = 2
        
        logging.info(f"| dictionary for CTC module: {len(dictionary)} types")
        logging.info(f"| CTC-based compression: {ctc_compression}")

    def forward(self, speech_out):        
        if "ctc_layer_result" in speech_out and speech_out["ctc_layer_result"] is not None:
            assert hasattr(self, "layer_norm")
            x = speech_out["ctc_layer_result"][0].transpose(0, 1)
        else:
            x = speech_out["encoder_out"][0]
            
        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)
            
        x = self.proj(self.dropout_module(x))
        
        return x.transpose(0, 1), {"attn": [], "inner_states": None}
    
    def compress(self, decoder_out, speech_out):

        x = speech_out["encoder_out"][0].transpose(0, 1) # T x B x D
        prev_lengths = speech_out["encoder_out_lengths"][0]
        lprobs_ctc = F.log_softmax(decoder_out[0], dim=-1).contiguous()  # T x B x V
        preds = torch.argmax(lprobs_ctc, dim=-1).contiguous()  # T x B
        
        _, B, D = x.size()
        x = x.transpose(0, 1)
        preds = preds.transpose(0, 1)
        x_compr = []
        preds_after_merged = []
        for i in range(B):
            p, c = preds[i].unique_consecutive(return_counts=True, dim=0)

            x_splt = torch.split(x[i], c.tolist())
            out = torch.stack([t.mean(dim=0) for t in x_splt])
            
            x_compr.append(out)
            preds_after_merged.append(p)

        x_compr = torch.nn.utils.rnn.pad_sequence(x_compr, batch_first=True)  # B x T x D
        preds_after_merged = torch.nn.utils.rnn.pad_sequence(
            preds_after_merged,
            batch_first=True,
            padding_value=self.blank_idx,
        )
        # Get mask of elements which are blank
        non_blank_mask = ~preds_after_merged.eq(self.blank_idx)
        # Get new lengths
        lengths = torch.sum(non_blank_mask, dim=-1)
        
        x_compr = x_compr.masked_select(non_blank_mask.unsqueeze(-1)).view(-1, D)
        x_compr = self._pad_seq_given_lens_arrays(x_compr, lengths)
        
        # need at least 2 frames for the length adaptor
        L = x_compr.size(1)
        if L < self.min_frames:
            x_compr = torch.cat([x_compr, torch.zeros(B, self.min_frames - L, D, device=x_compr.device)], dim=1)
        
        x_compr_mask = (torch.arange(x_compr.size(1), device=x_compr.device).expand(
            x_compr.size(0), -1) >= lengths.unsqueeze(1)).bool()
        
        decoder_out[-1]["compression_rate"] = (prev_lengths.float() - lengths.float()) / prev_lengths.float()

        assert x_compr.size(0) == speech_out["encoder_out"][0].size(0)
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [lengths]
        
        return decoder_out, speech_out
    
    def _pad_seq_given_lens_arrays(self, input, lengths, padding_value=0.0):
        cum_len = 0
        y = []
        for _, val in enumerate(lengths):
            y.append(input[cum_len : cum_len + val])
            cum_len += val
        return torch.nn.utils.rnn.pad_sequence(
            y, batch_first=True, padding_value=padding_value
        )

class SiameseSpeechTextEncoders(FairseqEncoder):
    def __init__(
        self,
        args,
        spch_encoder,
        dictionary,
        text_encoder=None,
        adaptor=None,
        context_encoder=None,
        bos_embedding=None,
        eos_embedding=None,
    ):
        super().__init__(dictionary)

        self.spch_encoder = spch_encoder
        
        if text_encoder is not None:
            self.text_encoder = text_encoder
        if adaptor is not None:
            self.adaptor = adaptor
        if context_encoder is not None:
            self.context_encoder = context_encoder
        if bos_embedding is not None:
            self.bos_embedding = bos_embedding
        if eos_embedding is not None:
            self.eos_embedding = eos_embedding

        self.ctc_layer_id = args.w2v_ctc_layer_id
        self.adaptor_after_context = args.adaptor_after_context
        self.freeze_text_encoder = args.freeze_text_encoder
        self.retain_dropout_in_frozen_text_encoder = args.retain_dropout_in_frozen_text_encoder
        self.freeze_speech_encoder_layers = args.freeze_speech_encoder_layers
        self.retain_dropout_in_frozen_speech_encoder = args.retain_dropout_in_frozen_speech_encoder

        self._maybe_freeze_text_encoder()  
        self._maybe_freeze_speech_encoder_layers()

    @classmethod
    def build_speech_encoder(cls, args):
    
        ckpt = torch.load(args.w2v_path)
        
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
        
        logger.info(f"Loading wav2vec2.0 encoder from {args.w2v_path} ...")
        missing_keys, unexpected_keys = spch_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        
        spch_encoder.embed_dim = spch_encoder.w2v_model.cfg.encoder_embed_dim
        spch_encoder.w2v_model.encoder.ctc_layer_id = args.w2v_ctc_layer_id
        
        args.w2v_cfg = w2v_model_config

        return spch_encoder, args

    @classmethod
    def build_text_encoder(cls, args, src_dictionary):
        ckpt = torch.load(args.mbart_path)
        
        if ckpt["args"] is None:
            model_args = ckpt["cfg"]["model"]
        else:
            model_args = ckpt["args"]
        
        enc_emb = build_embedding(src_dictionary, model_args.encoder_embed_dim)
        
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
        
        logger.info(f"Loading mBART encoder from {args.mbart_path} ...")
        missing_keys, unexpected_keys = text_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))

        return text_encoder
    
    @classmethod
    def build_context_encoder(cls, args, text_encoder):
        if not args.use_context_encoder:
            return None, args
        
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
        
        return context_encoder, args
    
    @classmethod
    def build_adaptor(cls, args, spch_encoder):
        if not args.conv1d_adaptor_layers:
            return None, args
        
        adaptor_cfg = Conv1dAdaptorConfig()
        adaptor_cfg.in_dim = spch_encoder.embed_dim
        adaptor_cfg.out_dim = spch_encoder.embed_dim
        adaptor_cfg.n_layers = args.conv1d_adaptor_layers
        adaptor_cfg.kernel_size = args.conv1d_adaptor_kernel_size
        adaptor_cfg.stride = args.conv1d_adaptor_stride
        
        adaptor = Conv1dAdaptor(adaptor_cfg)
        
        args.adaptor_args = adaptor_cfg
            
        return adaptor, args
    
    @classmethod
    def build_special_embeddings(cls, args, spch_encoder, text_encoder):
        eos_idx, bos_idx = 2, -49
        bos_emb, eos_emb = None, None
        
        if args.use_bos_embedding:
            if text_encoder is not None:
                assert text_encoder.dictionary.symbols[bos_idx] == "<lang:en>"
                weights = text_encoder.embed_tokens.weight[bos_idx].data
            else:
                weights = torch.zeros(spch_encoder.embed_dim)
            bos_emb = nn.Parameter(weights, requires_grad=True)
            
        if args.use_eos_embedding:
            if text_encoder is not None:
                assert text_encoder.dictionary.symbols[eos_idx] == "</s>"
                weights = text_encoder.embed_tokens.weight[eos_idx].data
            else:
                weights = torch.zeros(spch_encoder.embed_dim)
            eos_emb = nn.Parameter(weights, requires_grad=True)

        return bos_emb, eos_emb

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
            "encoder_out": [res["x"]],  # T x B x C
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
        
        x = speech_out[f"{key}_out"][0]
        if speech_out[f"{key}_padding_mask"][0] is not None:
            padding_mask = speech_out[f"{key}_padding_mask"][0].transpose(0, 1)
        else:
            padding_mask = None
        
        x = self.context_encoder(x, padding_mask)
        
        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["modified_out"] = [x]
        speech_out["modified_out_lengths"] = speech_out[f"{key}_out_lengths"]
        speech_out["modified_padding_mask"] = speech_out[f"{key}_padding_mask"]
        
        return speech_out
    
    def forward_text(self, src_txt_tokens, src_txt_lengths):
        if not hasattr(self, "text_encoder"):
            return None
        return self.text_encoder(src_txt_tokens, src_txt_lengths)
    
    def apply_special_embeddings(self, speech_out):
        if not hasattr(self, "bos_emb") and not hasattr(self, "eos_emb"):
            return speech_out
        
        if "modified_out" in speech_out:
            key = "modified"
        else:
            key = "encoder"
            
        x = speech_out[f"{key}_out"][0]
        lengths = speech_out[f"{key}_out_lengths"][0]
        if speech_out[f"{key}_padding_mask"][0] is not None:
            padding_mask = speech_out[f"{key}_padding_mask"][0].transpose(0, 1)
        else:
            padding_mask = None
        assert x.size(1) == lengths.size(0)
        B = x.size(1)
        
        if hasattr(self, "bos_emb"):
            x = torch.cat([self.bos_emb.unsqueeze(0).expand(1, B, -1), x], dim=1)
            lengths += 1
        if hasattr(self, "eos_emb"):
            x = torch.cat([x, self.eos_emb.unsqueeze(0).expand(1, B, -1)], dim=1)
            lengths += 1
    
        if padding_mask is not None:
            padding_mask = lengths_to_padding_mask(lengths)
        
        assert x.size(0) == speech_out[f"{key}_out"][0].size(0)
        speech_out["modified_out"] = [x]
        speech_out["modified_out_lengths"] = [lengths]
        speech_out["modified_padding_mask"] = [padding_mask]
        
        return speech_out
    
    def _maybe_freeze_text_encoder(self):
        if hasattr(self, "text_encoder") and self.freeze_text_encoder:
            logging.info(f"Freezing text encoder ...")
            for n, p in self.text_encoder.named_parameters():
                logging.info(f"- freezing {n}")
                p.requires_grad = False
                
    def _maybe_freeze_speech_encoder_layers(self):
        if self.freeze_speech_encoder_layers > -1:
            logging.info(f"Freezing speech encoder feature extractor ...")
            ft_layers = [
                ("feature_extractor", self.spch_encoder.w2v_model.feature_extractor),
                ("post_extract_proj", self.spch_encoder.w2v_model.post_extract_proj),
                ("pos_conv", self.spch_encoder.w2v_model.encoder.pos_conv)
            ]
            for name, layer in ft_layers:
                for n, p in layer.named_parameters():
                    logging.info(f"- freezing {name} {n}")
                    p.requires_grad = False
                
        if self.freeze_speech_encoder_layers > 0:
            logging.info(f"Freezing speech encoder layers ...")
            for i, layer in enumerate(self.spch_encoder.w2v_model.encoder.layers):
                if i < self.freeze_speech_encoder_layers:
                    for n, p in layer.named_parameters():
                        logging.info(f"- freezing layer{i} {n}")
                        p.requires_grad = False
                    if not self.retain_dropout_in_frozen_speech_encoder:
                        layer.self_attn.dropout_module.p = 0.0
                        layer.dropout1.p = 0.0
                        layer.dropout2.p = 0.0
                        layer.dropout3.p = 0.0

@register_model("siamese_st2t_transformer")
class SiameseST2TTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        # parser.add_argument(
        #     "--dropout", type=float, metavar="D", help="dropout probability"
        # )
        # parser.add_argument(
        #     "--activation-dropout", type=float, metavar="D", help="attention dropout probability"
        # )
        # parser.add_argument(
        #     "--attention-dropout", type=float, metavar="D", help="activation dropout probability"
        # )
        parser.add_argument(
            "--conv1d-adaptor-layers", type=int, default=0, help="number of layers in conv1d adaptor"
        )
        parser.add_argument(
            "--conv1d-adaptor-kernel-size", type=int, default=3, help="kernel size in conv1d adaptor"
        )
        parser.add_argument(
            "--conv1d-adaptor-stride", type=int, default=2, help="stride in conv1d adaptor"
        )
        parser.add_argument(
            "--ctc-compression", type=str, default="false",
            help="Apply CTC compression to speech encoder's output",
        )
        parser.add_argument(
            "--freeze-text-encoder", type=str, default="true", help="Freeze text encoder"
        )
        parser.add_argument(
            "--w2v-path", type=str, default="", help="path to pre-trained wav2vec model"
        )
        parser.add_argument(
            "--severed-layer",
            type=int,
            default=None,
            help="speech encoder layer to insert CTC module",
        ) 
        parser.add_argument(
            "--mbart-path", type=str, default="", help="path to pre-trained mbart model"
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
            help="use an mbart50 encoder on top of the speech encoder"
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
            "--adaptor-after-context", type=str, default="false",
            help="apply adaptor after context encoder, default is to apply before it"
        )
        parser.add_argument(
            "--freeze-speech-encoder-layers", type=int, default=-1,
            help="freeze the speech encoder layers with id < freeze-speech-encoder-layers"
            "-1 means no freezing, 0 means freeze feature-extractor, 1 means freeze feature-extractor and encoder.layer.0, etc..."
        )
        parser.add_argument(
            "--retain-dropout-in-frozen-speech-encoder", type=str, default="false",
            help="rerain dropout in frozen speech encoder layers"
        )
        parser.add_argument(
            "--no-text-encoder", type=str, default="false",
            help="do not use text encoder (no OT), use only speech encoder"
        )
        parser.add_argument(
            "--use-bos-embedding", type=str, default="false",
            help="use bos embedding in the output of the speech encoder"
        )
        parser.add_argument(
            "--use-eos-embedding", type=str, default="false",
            help="use eos embedding in the output of the speech encoder"
        )
        
    @classmethod
    def build_encoder(cls, args, task):
        spch_encoder, args = SiameseSpeechTextEncoders.build_speech_encoder(args)
        text_encoder = SiameseSpeechTextEncoders.build_text_encoder(args, task.src_dict)
        adaptor, args = SiameseSpeechTextEncoders.build_adaptor(args, spch_encoder)
        context_encoder, args = SiameseSpeechTextEncoders.build_context_encoder(args, text_encoder)
        bos_emb, eos_emb = SiameseSpeechTextEncoders.build_special_embeddings(args, spch_encoder, text_encoder)
        
        encoder = SiameseSpeechTextEncoders(
            args,
            spch_encoder,
            task.src_dict,
            text_encoder=text_encoder if not args.no_text_encoder else None,
            adaptor=adaptor,
            context_encoder=context_encoder,
            bos_embedding=bos_emb,
            eos_embedding=eos_emb,
        )
        
        return encoder

    @classmethod
    def build_decoder(cls, args, task, encoder):
        if args.ctc_weight > 0:
            decoder = CTCDecoder(
                task.target_dictionary,
                encoder.spch_encoder.embed_dim,
                encoder.spch_encoder.final_dropout.p,
                ctc_compression=args.ctc_compression,
                layernorm=(args.w2v_ctc_layer_id != -1),
            )
            # initialized the decoder's projection layer with the encoder's ctc projection layer
            decoder.proj = encoder.spch_encoder.proj
        else:
            decoder = DummyDecoder(task.target_dictionary)

        return decoder

    @classmethod
    def build_model(cls, args, task):
        
        default_bool = lambda x: getattr(args, x, "false") == "true"
        
        args.w2v_ctc_layer_id = getattr(args, "w2v_ctc_layer_id", -1)
        args.conv1d_adaptor_layers = getattr(args, "conv1d_adaptor_layers", 0)
        args.freeze_speech_encoder_layers = getattr(args, "freeze_speech_encoder_layers", -1)
        args.freeze_text_encoder = default_bool("freeze_text_encoder")
        args.adaptor_after_context = default_bool("adaptor_after_context")
        args.use_context_encoder = default_bool("use_context_encoder")
        args.retain_dropout_in_frozen_text_encoder = default_bool("retain_dropout_in_frozen_text_encoder")
        args.ctc_compression = default_bool("ctc_compression")
        args.w2v_apply_mask = default_bool("w2v_apply_mask")
        args.retain_dropout_in_frozen_speech_encoder = default_bool("retain_dropout_in_frozen_speech_encoder")
        args.no_text_encoder = default_bool("no_text_encoder")
        args.use_bos_embedding = default_bool("use_bos_embedding")
        args.use_eos_embedding = default_bool("use_eos_embedding")

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

    def forward(
        self,
        src_tokens=None,
        src_lengths=None,
        src_txt_tokens=None,
        src_txt_lengths=None,
    ):
        speech_out = self.encoder.forward_speech(src_tokens, src_lengths)
        
        decoder_out = None
        if isinstance(self.decoder, CTCDecoder):
            decoder_out = self.decoder(speech_out)
            
            if self.decoder.ctc_compression:
                decoder_out, speech_out = self.decoder.compress(decoder_out, speech_out)
                
        speech_out = self.encoder.apply_special_embeddings(speech_out)
        
        if not self.encoder.adaptor_after_context:
            speech_out = self.encoder.forward_adaptor(speech_out)
            
        speech_out = self.encoder.forward_context(speech_out)
        
        if self.encoder.adaptor_after_context:
            speech_out = self.encoder.forward_adaptor(speech_out)
        
        text_out = self.encoder.forward_text(src_txt_tokens, src_txt_lengths)
        
        return decoder_out, (speech_out, text_out)
    

@register_model_architecture("siamese_st2t_transformer", "siamese_st2t_transformer")
def siamese_st2t_transformer_base(args):
    pass