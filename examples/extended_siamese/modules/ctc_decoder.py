import logging
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqDecoder
from fairseq.modules import FairseqDropout, LayerNorm

logger = logging.getLogger(__name__)

@dataclass
class CTCBasedCompressionConfig(FairseqDataclass):
    type: str = field(
        default="letter",
        metadata={"help": "ctc-based compression type: letter or word"}
    )
    pooling_fn: str = field(
        default="mean",
        metadata={"help": "pooling function: max, mean, attention"}
    )


@dataclass
class CTCDecoderConfig(FairseqDataclass):
    embed_dim: int = field(
        default=1024,
        metadata={"help": "embedding dimension"}
    )
    dictionary_path: str = field(
        default=MISSING, metadata={"help": "path to the ctc model dictionary for inference"}
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout rate before the ctc projection layer"}
    )
    layernorm: bool = field(
        default=False,
        metadata={"help": "whether to use layer normalization before ctc projection layer"}
    )
    ctc_compression: Optional[CTCBasedCompressionConfig] = field(
        default=None,
        metadata={"help": "ctc-based compression config"}
    )

class CTCDecoder(FairseqDecoder):
    def __init__(self, cfg: CTCDecoderConfig):
        
        dictionary = Dictionary.load(cfg.dictionary_path)
        # correct the ctc dictionary
        dictionary.symbols[0], dictionary.symbols[1] = dictionary.symbols[1], dictionary.symbols[0]
        dictionary.indices["<s>"], dictionary.indices["<pad>"] = 1, 0
        dictionary.bos_index, dictionary.pad_index = 1, 0
    
        super().__init__(dictionary)

        self.cfg = cfg
        self.dictionary = dictionary
        self.blank_idx = dictionary.pad()  # TODO have to check for other encoders.
        self.sep_token = "|"
        self.sep_idx = dictionary.symbols.index(self.sep_token)

        # only if the expected input is not the final output of the speech encoder
        if cfg.layernorm:
            self.layer_norm = LayerNorm(cfg.embed_dim)

        self.dropout_module = FairseqDropout(cfg.dropout)
        self.proj = nn.Linear(cfg.embed_dim, len(dictionary), bias=True)

        logger.info(f"| dictionary for CTC module: {len(dictionary)} types")
        logger.info(f"| CTC-based compression: {cfg.ctc_compression}")

    def forward(self, speech_out):
        if (
            "ctc_layer_result" in speech_out
            and speech_out["ctc_layer_result"] is not None
        ):
            assert hasattr(self, "layer_norm")
            x = speech_out["ctc_layer_result"][0].transpose(0, 1)
        else:
            x = speech_out["encoder_out"][0]

        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)

        x = self.proj(self.dropout_module(x))

        return x.transpose(0, 1), {"attn": [], "inner_states": None}

    def compress(self, decoder_out, speech_out):
        # TODO move the overlapping functionality here
        if self.cfg.ctc_compression.type == "letter":
            return self.compress_letter(decoder_out, speech_out)
        elif self.cfg.ctc_compression.type == "word":
            return self.compress_word(decoder_out, speech_out)
        else:
            raise NotImplementedError

    def pool(self, x):
        if self.cfg.ctc_compression.pooling_fn == "mean":
            return torch.mean(x, dim=0)
        elif self.cfg.ctc_compression.pooling_fn == "max":
            return torch.max(x, dim=0)[0]
        elif self.cfg.ctc_compression.pooling_fn == "attention":
            raise NotImplementedError

    def compress_word(self, decoder_out, speech_out):
        x = speech_out["encoder_out"][0].transpose(0, 1)  # T x B x D
        prev_lengths = speech_out["encoder_out_lengths"][0]

        with torch.no_grad():
            lprobs_ctc = F.log_softmax(decoder_out[0], dim=-1)  # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1)  # T x B

        _, B, D = x.size()
        x = x.transpose(0, 1)  # B x T x D
        preds = preds.transpose(0, 1)  # B x T

        separetor_mask = preds == self.sep_idx
        valid_mask = preds != self.blank_idx

        t = valid_mask.sum(dim=-1).max().item()
        x_compr = torch.zeros(B, t, D, device=x.device, dtype=x.dtype)

        for i in range(B):
            if valid_mask[i].sum() == 0:
                continue
            x_i = x[i, valid_mask[i]]
            sep_indices = torch.nonzero(separetor_mask[i, valid_mask[i]]).squeeze()
            if sep_indices.dim() == 0:
                x_compr[i, 0] = self.pool(x_i)
            else:
                prev_idx = -1
                j = 0
                for idx in sep_indices:
                    if idx != prev_idx + 1:
                        x_compr[i, j] = self.pool(x_i[prev_idx + 1 : idx])
                        j += 1
                    prev_idx = idx

        lengths_compr = (x_compr.sum(dim=-1) != self.blank_idx).sum(dim=-1)  # B

        max_length = lengths_compr.max().item()
        if not max_length:
            return decoder_out, speech_out

        x_compr = x_compr[:, :max_length]
        x_compr_mask = lengths_to_padding_mask(lengths_compr)

        decoder_out[-1]["compression_rate"] = (
            prev_lengths.float() - lengths_compr.float()
        ) / prev_lengths.float()

        assert x_compr.size(0) == speech_out["encoder_out"][0].size(0)
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [lengths_compr]

        return decoder_out, speech_out
    
    def compress_letter(self, decoder_out, speech_out):
        x = speech_out["encoder_out"][0].transpose(0, 1)  # T x B x D
        prev_lengths = speech_out["encoder_out_lengths"][0] # B

        with torch.no_grad():
            lprobs_ctc = F.log_softmax(decoder_out[0], dim=-1)  # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1)  # T x B

        T, B, D = x.size()
        x = x.transpose(0, 1) # B x T x D
        preds = preds.transpose(0, 1) # B x T

        # max number of non-blank tokens in a batch of sequences
        N = (preds != self.blank_idx).sum(dim=-1).max().item()
        if not N:
            N = 1 # for rare case where whole batch is blank
        
        # create the compressed sequence with N positions
        # will be further reduced after getting the actual lengths
        x_compr = torch.zeros(B, N, D, device=x.device, dtype=x.dtype) # B x N x D
        lengths_compr = torch.zeros(B, device=x.device, dtype=torch.long) # B
        valid_examples = torch.zeros(B, device=x.device, dtype=torch.bool) # B

        for i in range(B):
            # p: compressed sequence, [N_i]
            # c: repeating counts for each j element of p, [N_i]
            p, c = preds[i].unique_consecutive(return_counts=True)
            valid_mask_i = p != self.blank_idx # [N_i]
            x_splt = torch.split(x[i], c.tolist()) # Tuple[tensor] of size N_i, x_splt[0] is the first c[0] elements of x[i]
            out = torch.stack([self.pool(t) for t in x_splt]) # N_i x D
            if not valid_mask_i.any():
                # empty examples have just one blank
                x_compr[i, :1] = out
                lengths_compr[i] = 1
            else:
                # remove blank tokens
                out = out[valid_mask_i] # N_i' x D

                x_compr[i, :out.size(0)] = out
                lengths_compr[i] = out.size(0)
                valid_examples[i] = True

        # real max length after removing blanks
        max_length = lengths_compr.max().item()

        x_compr = x_compr[:, :max_length]
        x_compr_mask = lengths_to_padding_mask(lengths_compr)
        
        decoder_out[-1]["compression_rate"] = (
            prev_lengths.float() - lengths_compr.float()
        ) / prev_lengths.float()
        speech_out["modified_valid_examples"] = [valid_examples]
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [lengths_compr]

        return decoder_out, speech_out