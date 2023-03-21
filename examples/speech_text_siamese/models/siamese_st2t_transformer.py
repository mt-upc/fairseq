#!/usr/bin/env python3

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import S2TTransformerEncoder
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.models.transformer import Embedding, TransformerEncoder
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.modules import FairseqDropout, LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout

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
    def __init__(self, dictionary, embed_dim, task, dropout_rate=0.0, bias=True):
        super().__init__(dictionary)
        self.blank_idx = (
            dictionary.index(task.blank_symbol) if hasattr(task, "blank_symbol") else 0
        )
        self.pad_idx = dictionary.pad()
        self.eos_idx = dictionary.eos()
        self.dropout_module = FairseqDropout(dropout_rate)
        self.proj = Linear(embed_dim, len(dictionary), bias=bias)
        logging.info(f"| dictionary for CTC module: {len(dictionary)} types")

    def forward(
        self,
        encoder_out: Optional[Dict[str, List[Tensor]]],
    ):
        if "severed_encoder_out" not in encoder_out:  # backward comptability
            x = encoder_out["encoder_out"][0]
        else:
            if encoder_out["severed_encoder_out"][0] is not None:
                x = encoder_out["severed_encoder_out"][0]
            else:
                x = encoder_out["encoder_out"][0]
        x = x.transpose(0, 1)  # B x T x D
        x = self.proj(self.dropout_module(x))
        return x.transpose(0, 1), {"attn": [], "inner_states": None}


class DummyDecoder(FairseqDecoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)


class SpeechEncoderWithAdapter(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)
        self.spch_encoder = S2TTransformerEncoder(args)
        self.cnn_module = None
        if not getattr(args, "no_cnn_in_adapter", False):
            self.cnn_module = Conv1dSubsampler(
                args.encoder_embed_dim,
                args.conv_channels,
                args.encoder_embed_dim,
                [3],
            )
        self.fc = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim),
            LayerNorm(args.encoder_embed_dim),
        )

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        speech_out = self.spch_encoder(
            src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
        )
        # logging.info(f"speech_out: {speech_out['encoder_out'][0].size()}") # T x B x D
        # logging.info(f"input_lengths: {speech_out['input_lengths'][0].size()}") # B
        x = speech_out["encoder_out"][0]
        if speech_out["encoder_padding_mask"]:
            encoder_padding_mask = speech_out["encoder_padding_mask"][0]
        else:
            encoder_padding_mask = torch.zeros(1)
        input_lens = speech_out["input_lengths"][0]
        if self.cnn_module is not None:
            x, input_lens = self.cnn_module(
                speech_out["encoder_out"][0].transpose(0, 1),
                speech_out["input_lengths"][0],
            )
            encoder_padding_mask = lengths_to_padding_mask(input_lens)
        # logging.info(f"x: {x.size()}, input_lens: {input_lens.size()}, encoder_padding_mask: {encoder_padding_mask.size()}")
        x = self.fc(x)
        # logging.info(f"x: {x.size()}")

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "input_lengths": [input_lens],
        }


class Wav2VecEncoderWithTransformer(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)
        self.freeze_w2v_encoder = args.freeze_w2v_encoder
        ckpt = torch.load(args.w2v_path)
        w2v_args = ckpt["args"]
        w2v_model_config = convert_namespace_to_omegaconf(w2v_args).model
        self.w2v_encoder = Wav2Vec2Model(w2v_model_config)
        self.w2v_encoder.load_state_dict(ckpt["model"])
        args = args._replace(input_feat_per_channel=w2v_args.encoder_embed_dim)
        self.transformer_encoder = S2TTransformerEncoder(args)

    def _get_w2v_feature(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        res = self.w2v_encoder.extract_features(src_tokens, padding_mask)
        padding_mask = res["padding_mask"]
        if padding_mask is not None:
            output_lengths = (1 - padding_mask.int()).sum(dim=1)
        else:
            B, T, _ = res["x"].size()
            output_lengths = (torch.ones(B, device=res["x"].device) * T).long()
        return res["x"], output_lengths

    def forward(self, src_tokens, src_lengths):
        x, input_lengths = self._get_w2v_feature(src_tokens, src_lengths)
        x = self.transformer_encoder(x, input_lengths)
        return x


class SiameseSpeechTextEncoders(FairseqEncoder):
    def __init__(
        self,
        args,
        spch_encoder,
        dictionary,
        text_encoder,
    ):
        super().__init__(dictionary)

        self.spch_encoder = spch_encoder
        self.text_encoder = text_encoder
        self.shrink_speech_output = getattr(args, "shrink_speech_output", False)
        self.zero_speech_output = getattr(args, "zero_speech_output", False)

    @classmethod
    def build_speech_encoder(cls, args):
        cfg = {
            "input_feat_per_channel": args.input_feat_per_channel,
            "input_channels": getattr("args", "input_channels", 1),
            "conv_kernel_sizes": args.conv_kernel_sizes,
            "conv_channels": args.conv_channels,
            "encoder_embed_dim": args.encoder_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.speech_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "max_source_positions": args.max_source_positions,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "layernorm_embedding": args.layernorm_embedding,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
            "encoder_freezing_updates": 0,
            "no_cnn_in_adapter": getattr(args, "no_cnn_in_adapter", False),
            "freeze_w2v_encoder": getattr(args, "freeze_w2v_encoder", False),
            "w2v_path": getattr(args, "w2v_path", ""),
            "severed_layer": getattr(args, "severed_layer", None),
        }
        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        if getattr(args, "use_w2v_encoder", False):
            spch_encoder = Wav2VecEncoderWithTransformer(model_args)
        elif not getattr(args, "speech_encoder_with_adapter", False):
            spch_encoder = S2TTransformerEncoder(model_args)
        else:
            spch_encoder = SpeechEncoderWithAdapter(model_args)
        return spch_encoder

    @classmethod
    def build_text_encoder(cls, args, src_dictionary):
        cfg = {
            "encoder_embed_dim": args.encoder_text_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.text_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "max_source_positions": args.max_positions_text,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "adaptive_input": args.adaptive_input,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
            "layernorm_embedding": args.encoder_text_layernorm_embedding,
        }
        text_encoder = None

        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        enc_emb = build_embedding(src_dictionary, model_args.encoder_embed_dim)
        text_encoder = TransformerEncoder(model_args, src_dictionary, enc_emb)

        return text_encoder

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
        # src_tokens only: inference
        # src_tokens, src_lengths: speech only training
        # src_tokens, src_txt_tokens, src_txt_lengths: siamese training

        # if src_tokens is None and src_txt_tokens is None and masked_src_txt_tokens is None:
        #     raise ValueError(
        #         "src_tokens and src_txt_tokens and masked_src_txt_tokens cannot be None at the same time"
        #     )
        # if src_tokens is not None and self.spch_encoder is not None:
        speech_out = self.spch_encoder(src_tokens, src_lengths)
        # if src_txt_tokens is not None and self.text_encoder is not None:
        text_out = self.text_encoder(src_txt_tokens, src_txt_lengths)

        return speech_out, text_out

    def reorder_encoder_out(self, encoder_out, new_order):
        assert self.training is False  # used for inference only
        if self.spch_encoder is not None:
            if hasattr(self.spch_encoder, "w2v_encoder"):
                return self.spch_encoder.transformer_encoder.reorder_encoder_out(
                    encoder_out, new_order
                )
            return self.spch_encoder.reorder_encoder_out(encoder_out, new_order)
        else:
            return self.text_encoder.reorder_encoder_out(encoder_out, new_order)


@register_model("siamese_st2t_transformer")
class SiameseST2TTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # encoder 1: S2TTransformerEncoder for speech
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # standard Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-text-embed-dim",
            type=int,
            metavar="N",
            help="encoder text embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        # non-standard transformer parameters
        parser.add_argument(
            "--speech-encoder-layers",
            type=int,
            metavar="N",
            help="num speech encoder layers",
        )
        parser.add_argument(
            "--text-encoder-layers",
            type=int,
            metavar="N",
            help="num text encoder layers",
        )
        parser.add_argument(
            "--load-pretrain-speech-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help="path to the pretrained speech encoder",
        )
        parser.add_argument(
            "--load-pretrain-text-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help="path to the pretrained text encoder",
        )
        parser.add_argument(
            "--freeze-text-encoder-embed",
            action="store_true",
            help="Freeze embedding layer of text encoder",
        )
        parser.add_argument(
            "--num-text-encoder-layers-frozen",
            type=int,
            default=0,
            help="Number of layers to be frozen in text encoder",
        )
        # additional parameters for Siamese encoders
        parser.add_argument(
            "--use-ctc-module",
            action="store_true",
            help="Use CTC module (Linear + Softmax) after the encoder",
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
            "--share-text-encoder-ctc-decoder-input-output",
            action="store_true",
            help="share text encoder embed and ctc output layer",
        )
        parser.add_argument(
            "--encoder-text-layernorm-embedding",
            action="store_true",
            help="add layernorm to text encoder embedding",
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
        ) # TODO: add this for wav2vec

    @classmethod
    def build_encoder(cls, args, task):
        spch_encoder = SiameseSpeechTextEncoders.build_speech_encoder(args)
        text_encoder = SiameseSpeechTextEncoders.build_text_encoder(args, task.src_dict)
        encoder = SiameseSpeechTextEncoders(
            args,
            spch_encoder,
            task.src_dict,
            text_encoder=text_encoder,
        )
        if getattr(args, "load_pretrain_speech_encoder", "") != "":
            logging.info(f"Loading pretrained speech encoder ...")
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.load_pretrain_speech_encoder
            )
            ckpt_component_type = (
                ["encoder.spch_encoder"]
                if any(
                    [
                        key.startswith("encoder.spch_encoder")
                        for key in state["model"].keys()
                    ]
                )
                else ["encoder"]
            )
            checkpoint_utils.load_pretrained_component_from_model_different_keys(
                spch_encoder, state, ckpt_component_types=ckpt_component_type
            )
            logging.info(
                f"Loaded pretrained speech encoder from {args.load_pretrain_speech_encoder}"
            )

        if getattr(args, "load_pretrain_text_encoder", "") != "":
            logging.info(f"Loading pretrained text encoder ...")
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.load_pretrain_text_encoder
            )
            ckpt_component_type = (
                ["decoder.sentence_encoder"]
                if any(
                    [
                        key.startswith("decoder.sentence_encoder")
                        for key in state["model"].keys()
                    ]
                )
                else ["encoder"]
            )
            checkpoint_utils.load_pretrained_component_from_model_different_keys_v2(
                text_encoder,
                state,
                ckpt_component_types=ckpt_component_type,
                # exclude_layers=["embed_tokens", "embed_positions", "emb_layer_norm"]
            )
            logging.info(
                f"Loaded pretrained text encoder from {args.load_pretrain_text_encoder}"
            )

        if getattr(args, "use_w2v_encoder", False) and getattr(
            args, "freeze_w2v_encoder", False
        ):
            logging.info(f"Freezeing wav2vec encoder ...")
            for n, p in spch_encoder.w2v_encoder.named_parameters():
                logging.info(f"- freezing {n}")
                p.requires_grad = False

        if getattr(args, "freeze_text_encoder", False):
            logging.info(f"Freezing text encoder ...")
            for n, p in text_encoder.named_parameters():
                logging.info(f"- freezing {n}")
                p.requires_grad = False

        if getattr(args, "freeze_text_encoder_embed", False):
            logging.info(f"Freezing text encoder embedding layer...")
            text_encoder.embed_tokens.requires_grad = False
            text_encoder.embed_positions.requires_grad = False

        if getattr(args, "num_text_encoder_layers_frozen", 0) > 0:
            logging.info(f"Freezing text encoder transformer layer...")
            for l in range(-args.num_text_encoder_layers_frozen, 0, 1):
                logging.info(f"- freezing layer {l + args.text_encoder_layers}...")
                text_encoder.layers[l].requires_grad = False

        return encoder

    @classmethod
    def build_decoder(cls, args, task, encoder):
        if args.use_ctc_module:
            decoder = CTCDecoder(
                task.source_dictionary,
                args.decoder_embed_dim,
                task,
                args.dropout,
                bias=not getattr(args, "no_bias_in_proj", False),
            )
            if getattr(args, "share_text_encoder_ctc_decoder_input_output", False):
                assert encoder.text_encoder is not None
                encoder.text_encoder.embed_tokens.weight = decoder.proj.weight
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
    "siamese_st2t_transformer", "siamese_st2t_transformer_base"
)
def siamese_st2t_transformer_base(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_text_embed_dim = getattr(
        args, "encoder_text_embed_dim", args.encoder_embed_dim
    )
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_text_layernorm_embedding = getattr(
        args, "encoder_text_layernorm_embedding", False
    )

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)

    args.use_ctc_module = getattr(args, "use_ctc_module", True)


@register_model_architecture(
    "siamese_st2t_transformer", "siamese_st2t_transformer_s"
)
def siamese_st2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_text_embed_dim = getattr(args, "encoder_text_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    siamese_st2t_transformer_base(args)


@register_model_architecture(
    "siamese_st2t_transformer", "siamese_st2t_transformer_m"
)
def siamese_st2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    siamese_st2t_transformer_base(args)


@register_model_architecture(
    "siamese_st2t_transformer", "siamese_st2t_transformer_l"
)
def siamese_st2t_transformer_l(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.25)
    siamese_st2t_transformer_base(args)
