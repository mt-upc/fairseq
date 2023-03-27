#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
from fairseq.modules.length_adaptor import Conv1dAdaptor

logger = logging.getLogger(__name__)


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


class CTCDecoder(FairseqDecoder):
    def __init__(self, dictionary, embed_dim, task, dropout_rate=0.0, bias=True, layernorm=False):
        super().__init__(dictionary)
        self.blank_idx = (
            dictionary.index(task.blank_symbol) if hasattr(task, "blank_symbol") else 0
        )
        self.pad_idx = dictionary.pad()
        self.eos_idx = dictionary.eos()
        if layernorm:
            self.layer_norm = LayerNorm(embed_dim)
        self.dropout_module = FairseqDropout(dropout_rate)
        self.proj = Linear(embed_dim, len(dictionary), bias=bias)
        logging.info(f"| dictionary for CTC module: {len(dictionary)} types")

    def forward(
        self,
        encoder_out: Optional[Dict[str, List[Tensor]]],
    ):
        if "ctc_layer_result" in encoder_out and encoder_out["ctc_layer_result"] is not None:
            assert hasattr(self, "layer_norm")
            x = encoder_out["ctc_layer_result"][0].transpose(0, 1)
        else:
            x = encoder_out["encoder_out"][0]
            
        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)
            
        x = self.proj(self.dropout_module(x))
        
        return x.transpose(0, 1), {"attn": [], "inner_states": None}


class DummyDecoder(FairseqDecoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)

class Wav2VecMbartEncoder(FairseqEncoder):
    pass


class SiameseSpeechTextEncoders(FairseqEncoder):
    def __init__(
        self,
        args,
        spch_encoder,
        dictionary,
        text_encoder,
        adaptor=None
    ):
        super().__init__(dictionary)

        self.spch_encoder = spch_encoder
        self.text_encoder = text_encoder
        if adaptor is not None:
            self.adaptor = adaptor
        self.shrink_speech_output = getattr(args, "shrink_speech_output", False)
        self.zero_speech_output = getattr(args, "zero_speech_output", False)
        self.freeze_text_encoder = getattr(args, "freeze_text_encoder", False)
        self.retain_dropout_in_frozen_text_encoder = \
            getattr(args, "retain_dropout_in_frozen_text_encoder", False)
        self.ctc_layer_id = getattr(args, "w2v_ctc_layer_id", -1)

    @classmethod
    def build_speech_encoder(cls, args):
    
        ckpt = torch.load(args.w2v_path)
        
        w2v_args = ckpt["args"]
        w2v_model_config = convert_namespace_to_omegaconf(w2v_args).model      
        
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
        
        spch_encoder.freeze_finetune_updates = 0
        spch_encoder.w2v_model.encoder.layerdrop = args.w2v_layerdrop
        spch_encoder.embed_dim = spch_encoder.w2v_model.cfg.encoder_embed_dim
        spch_encoder.w2v_model.encoder.ctc_layer_id = args.w2v_ctc_layer_id

        return spch_encoder

    @classmethod
    def build_text_encoder(cls, args, src_dictionary):       
        ckpt = torch.load(args.mbart_path)
        
        enc_emb = build_embedding(src_dictionary, ckpt["args"].encoder_embed_dim)
        
        text_encoder = TransformerEncoder(
            ckpt["args"], src_dictionary, enc_emb
        )
        
        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("encoder."):
                model_ckpt[k.replace("encoder.", "")] = v
        
        logger.info(f"Loading mBART encoder from {args.mbart_path} ...")
        missing_keys, unexpected_keys = text_encoder.load_state_dict(model_ckpt, strict=False)
        if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
        if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        
        if getattr(args, "freeze_text_encoder", False):
            logging.info(f"Freezing text encoder ...")
            for n, p in text_encoder.named_parameters():
                logging.info(f"- freezing {n}")
                p.requires_grad = False
                
            if not getattr(args, "retain_dropout_in_frozen_text_encoder", False):
                text_encoder.eval()

        return text_encoder
    
    @classmethod
    def build_adaptor(cls, args, spch_encoder):
        if getattr(args, "conv1d_adaptor_layers", 0) > 0:
            adaptor = Conv1dAdaptor(
                in_dim=spch_encoder.embed_dim,
                out_dim=spch_encoder.embed_dim,
                n_layers=args.conv1d_adaptor_layers,
                kernel_size=args.conv1d_adaptor_kernel_size,
                stride=args.conv1d_adaptor_stride
            )
        else:
            adaptor = None
        return adaptor

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        src_txt_tokens=None,
        src_txt_lengths=None,
        
    ):
        """
        Args:
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (speech) (B,)
            src_txt_tokens: padded tensor (B, T)
            src_txt_lengths: tensor of original lengths of input utterances (text) (B,)
        """
        speech_out = self.forward_speech(src_tokens, src_lengths)
        if hasattr(self, "adaptor"):
            speech_out = self.forward_adaptor(speech_out)
        text_out = self.forward_text(src_txt_tokens, src_txt_lengths)

        return speech_out, text_out
    
    def forward_adaptor(self, speech_out):

        x, padding_mask = self.adaptor(
            speech_out["encoder_out"][0].transpose(0, 1),
            speech_out["encoder_padding_mask"][0] if speech_out["encoder_padding_mask"][0] is not None else None,
            )
        x = x.transpose(0, 1)
        
        if padding_mask is not None:
            output_lengths = (1 - padding_mask.int()).sum(dim=1)
        else:
            B, T, _ = x.size()
            output_lengths = (torch.ones(B, device=x.device) * T).long()
        
        speech_out["adaptor_out"] = [x]
        speech_out["adaptor_padding_mask"] = [padding_mask]
        speech_out["adaptor_out_lengths"] = [output_lengths]
        
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
    
    def forward_text(self, src_txt_tokens, src_txt_lengths):
        if self.freeze_text_encoder and not self.retain_dropout_in_frozen_text_encoder:
            with torch.no_grad():
                return self.text_encoder(src_txt_tokens, src_txt_lengths)
        return self.text_encoder(src_txt_tokens, src_txt_lengths)


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
            "--shrink-speech-output",
            action="store_true",
            help="Shrink speech encoder's output based on CTC module",
        )
        parser.add_argument(
            "--zero-speech-output",
            action="store_true",
            help="Zero out speech encoder's output based on CTC module",
        )
        parser.add_argument(
            "--freeze-text-encoder", action="store_true", help="Freeze text encoder"
        )
        parser.add_argument(
            "--speech-encoder-with-adapter",
            action="store_true",
            help="use speech encoder with adapter",
        )
        parser.add_argument(
            "--no-cnn-in-adapter", action="store_true", help="no CNN module in adapter"
        )
        parser.add_argument(
            "--use-w2v-encoder", action="store_true", help="use wav2vec encoder"
        )
        parser.add_argument(
            "--freeze-w2v-encoder", action="store_true", help="freeze wav2vec encoder"
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
            "--use-mbart-text-encoder", action="store_true", help="use mbart text encoder"
        )
        parser.add_argument(
            "--mbart-path", type=str, default="", help="path to pre-trained mbart model"
        )
        parser.add_argument(
            "--retain-dropout-in-frozen-text-encoder", action="store_true", help="retain dropout in frozen mbart text encoder"
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
            "--w2v-apply-mask", action="store_true", help="apply mask to wav2vec speech encoder"
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

    @classmethod
    def build_encoder(cls, args, task):
        spch_encoder = SiameseSpeechTextEncoders.build_speech_encoder(args)
        text_encoder = SiameseSpeechTextEncoders.build_text_encoder(args, task.src_dict)
        adaptor = SiameseSpeechTextEncoders.build_adaptor(args, spch_encoder)
        encoder = SiameseSpeechTextEncoders(
            args,
            spch_encoder,
            task.src_dict,
            text_encoder=text_encoder,
            adaptor=adaptor
        )
        
        return encoder

    @classmethod
    def build_decoder(cls, args, task, encoder):
        if args.ctc_weight > 0:
            decoder = CTCDecoder(
                task.source_dictionary,
                encoder.spch_encoder.embed_dim,
                task,
                encoder.spch_encoder.final_dropout.p,
                layernorm=getattr(args, "w2v_ctc_layer_id", -1) != -1,
            )
            
            # initialized the decoder's projection layer with the encoder's ctc projection layer
            decoder.proj = encoder.spch_encoder.proj
            
        else:
            decoder = DummyDecoder(task.source_dictionary)

        return decoder

    @classmethod
    def build_model(cls, args, task):
        # torch.autograd.set_detect_anomaly(True)
        # make sure that all args are properly defaulted
        siamese_st2t_transformer_base(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task, encoder)
        
        # do it after initializing the decoder to transfer the ctc weights
        encoder.spch_encoder.w2v_model.remove_pretraining_modules()
        
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None, idx=0):
        lprobs = self.get_normalized_probs_scriptable(
            net_output, log_probs, sample, idx=idx
        )
        lprobs.batch_first = True
        return lprobs

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        idx=0,
    ):
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
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_txt_tokens=src_txt_tokens,
            src_txt_lengths=src_txt_lengths,
        )
        if isinstance(self.decoder, DummyDecoder):
            return None, encoder_out
        elif isinstance(self.decoder, CTCDecoder):
            decoder_out = self.decoder(
                encoder_out=encoder_out[0]
                if isinstance(encoder_out, tuple)
                else encoder_out,
            )
        else:
            raise NotImplementedError

        def zero_speech_output(speech_out, preds):
            """
            Zero elements in x (corresponding to repeated consecutive
                                values or blank index in preds)
            Args:
                x: T x B x D
                preds: T x B
            """
            preds = preds.double()
            D = speech_out.size()[-1]
            T, B = preds.size()
            # get indices to be removed (blank tokens) and merge repeated predictions into 1
            # construct a difference matrix having below format
            # [[ 1,  0,  0,  0,  ...],
            #  [-1,  1,  0,  0,  ...],
            #  [ 0, -1,  1,  0,  ...],
            #  [ 0,  0, -1,  1,  ...],
            diff_matrix = (
                (torch.triu(torch.tril(torch.ones(T, T) * -1), -1) + torch.eye(T) * 2)
                .to(preds.device)
                .double()
            )  # T x T
            diff_preds = torch.matmul(diff_matrix, preds)  # T x B
            blank_idx = (
                self.decoder.blank_idx
                if isinstance(self.decoder, CTCDecoder)
                else self.decoder.ctc_module.blank_idx
            )
            m = ~(preds.eq(blank_idx) | diff_preds.eq(0))  # T x B
            reduced_t = T * B - torch.sum(m)
            m = m.transpose(0, 1).unsqueeze(2).expand(-1, -1, D)  # B x T x D
            speech_out = speech_out.transpose(0, 1) * m  # B x T x D
            return speech_out.transpose(0, 1), reduced_t / (T * B)

        def pad_seq_given_lens_arrays(input, lengths, padding_value=0.0):
            """
            Reshape and pad an input tensor given lengths of each chunk in the tensor
            """
            cum_len = 0
            y = []
            for _, val in enumerate(lengths):
                y.append(input[cum_len : cum_len + val])
                cum_len += val
            return torch.nn.utils.rnn.pad_sequence(
                y, batch_first=True, padding_value=padding_value
            )

        def shrink_speech_output(speech_out, preds):
            """
            Average elements in x correponsing to repeated consecutive values in preds
            Args:
                x: T x B x D
                preds: T x B
            """
            # iterate through batch dimension
            T, B, D = speech_out.size()
            speech_out = speech_out.transpose(0, 1)
            preds = preds.transpose(0, 1)
            Y = []
            preds_after_merged = []
            reduced_t = 0
            for i in range(B):
                p, c = preds[i].unique_consecutive(return_counts=True, dim=0)
                # create a padded tensor of shape num_chunks x max_len_chunks x D
                padded = pad_seq_given_lens_arrays(speech_out[i], c)  # N x S x D
                # sum over each chunk and divide by lengths
                out = torch.sum(padded, dim=1) / c.unsqueeze(-1).expand(-1, D)
                Y.append(out)
                preds_after_merged.append(p)
                reduced_t += torch.sum(c[~c.eq(1)]) - torch.numel(c[~c.eq(1)])

            Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)  # B x T x D
            preds_after_merged = torch.nn.utils.rnn.pad_sequence(
                preds_after_merged,
                batch_first=True,
                padding_value=self.decoder.pad_idx,
            )
            # Get mask of elements which are blank
            non_blank_mask = ~preds_after_merged.eq(self.decoder.blank_idx)
            # if preds_after_merged are all blank then not reducing
            non_blank_mask = (
                ~non_blank_mask if torch.all(~non_blank_mask) else non_blank_mask
            )
            reduced_t += torch.sum(~non_blank_mask)
            # Get new lengths
            lengths = torch.sum(non_blank_mask, dim=-1)
            Y = Y.masked_select(non_blank_mask.unsqueeze(-1)).view(-1, D)
            Y = pad_seq_given_lens_arrays(Y, lengths)
            return Y.transpose(0, 1), reduced_t / (T * B)

        assert isinstance(self.decoder, CTCDecoder)
        speech_out = decoder_out[0]  # T x B x V
        x = speech_out

        # Shrink speech output
        if self.encoder.shrink_speech_output or self.encoder.zero_speech_output:
            assert isinstance(self.decoder, CTCDecoder)
            ctc_out = (
                decoder_out[0]
                if isinstance(self.decoder, CTCDecoder)
                else decoder_out[0][1]
                if isinstance(decoder_out[0], tuple)
                else decoder_out[0]
            )
            lprobs_ctc = F.log_softmax(ctc_out, dim=-1).contiguous()  # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1).contiguous()  # T x B

            if self.encoder.zero_speech_output:
                x, reduced_t = zero_speech_output(speech_out, preds)
            elif self.encoder.shrink_speech_output:
                x, reduced_t = shrink_speech_output(speech_out, preds)
            else:
                raise NotImplementedError

            decoder_out[-1]["reduced_speech_output"] = reduced_t

            if isinstance(encoder_out, tuple):
                encoder_out[0]["encoder_out"] = [x]  # T x B x D or T x B x V
            else:
                encoder_out["encoder_out"] = [x]  # T x B x D or T x B x V

        return decoder_out, encoder_out


@register_model_architecture(
    "siamese_st2t_transformer", "siamese_st2t_transformer_w2v_mbart"
)
def siamese_st2t_transformer_base(args):
    args.use_w2v_encoder = getattr(args, "use_w2v_encoder", True)
    args.use_mbart_text_encoder = getattr(args, "use_mbart_text_encoder", True)
    args.freeze_text_encoder = getattr(args, "freeze_text_encoder", True)