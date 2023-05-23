import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqDecoder
from fairseq.modules import FairseqDropout, LayerNorm

logger = logging.getLogger(__name__)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    logger.info(f"| bias in Linear layer: {bias}")
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@dataclass
class CTCDecoderConfig(FairseqDataclass):
    embed_dim: int = field(default=1024, metadata={"help": "embedding dimension"})
    dropout_rate: float = field(
        default=0.0, metadata={"help": "dropout rate before the ctc projection layer"}
    )
    ctc_compression: bool = field(
        default=False, metadata={"help": "whether to use ctc-based compression"}
    )
    ctc_compression_type: str = field(
        default="letter",
        metadata={"help": "ctc-based compression type: letter or word"},
    )
    post_compression_layer: bool = field(
        default=False, metadata={"help": "whether to use post-compression layer"}
    )
    post_compression_dim: int = field(
        default=4096, metadata={"help": "post-compression layer dimension"}
    )
    pooling_fn: str = field(
        default="mean", metadata={"help": "pooling function: max, mean, attention"}
    )
    layernorm: bool = field(
        default=False,
        metadata={
            "help": "whether to use layer normalization before ctc projection layer"
        },
    )
    path: str = field(
        default="", metadata={"help": "path to pre-trained ctc decoder checkpoint"}
    )
    dictionary: str = field(
        default="", metadata={"help": "path to the ctc model dictionary for inference"}
    )
    final_layernorm: bool = field(
        default=False, metadata={"help": "whether to use final layer normalization"}
    )
    freeze: bool = field(
        default=False, metadata={"help": "freeze ctc decoder parameters"}
    )


class CTCDecoder(FairseqDecoder):
    def __init__(self, dictionary, cfg: CTCDecoderConfig):
        super().__init__(dictionary)

        self.cfg = cfg
        self.dictionary = dictionary
        self.blank_idx = dictionary.pad()  # TODO have to check for other encoders.
        self.sep_token = "|"
        self.sep_idx = dictionary.symbols.index(self.sep_token)

        # only if the expected input is not the final output of the speech encoder
        if cfg.layernorm:
            self.layer_norm = LayerNorm(cfg.embed_dim)

        self.dropout_module = FairseqDropout(cfg.dropout_rate)
        self.proj = Linear(cfg.embed_dim, len(dictionary), bias=True)

        logger.info(f"| dictionary for CTC module: {len(dictionary)} types")
        logger.info(f"| CTC-based compression: {cfg.ctc_compression}")
        logger.info(f"| CTC-based compression type: {cfg.ctc_compression_type}")

        if cfg.ctc_compression and cfg.post_compression_layer:
            self.post = nn.Sequential(
                LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim,  cfg.post_compression_dim),
                nn.GELU(),
                nn.Linear(cfg.post_compression_dim, cfg.embed_dim),
            )
            logger.info(f"| post-compression layer: True with dim {cfg.post_compression_dim}")
        if cfg.ctc_compression and cfg.final_layernorm:
            self.final_layernorm = LayerNorm(cfg.embed_dim)

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
        if self.cfg.ctc_compression_type == "letter":
            return self.compress_letter(decoder_out, speech_out)
        elif self.cfg.ctc_compression_type == "word":
            return self.compress_word(decoder_out, speech_out)
        else:
            raise NotImplementedError

    def pool(self, x):
        if self.cfg.pooling_fn == "mean":
            return torch.mean(x, dim=0)
        elif self.cfg.pooling_fn == "max":
            return torch.max(x, dim=0)[0]
        elif self.cfg.pooling_fn == "attention":
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

        if hasattr(self, "post"):
            x_compr = self.post(x_compr)
                
        if hasattr(self, "final_layernorm"):
            x_compr = self.final_layernorm(x_compr)

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
        prev_lengths = speech_out["encoder_out_lengths"][0]

        with torch.no_grad():
            lprobs_ctc = F.log_softmax(decoder_out[0], dim=-1)  # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1)  # T x B

        _, B, D = x.size()
        x = x.transpose(0, 1)
        preds = preds.transpose(0, 1)

        t = (preds != self.blank_idx).sum(dim=-1).max().item()
        if not t:
            t = 1 # for rare case where whole batch is blank
        x_compr = torch.zeros(B, t, D, device=x.device, dtype=x.dtype)
        lengths_compr = torch.zeros(B, device=x.device, dtype=torch.long)
        valid_examples = torch.zeros(B, device=x.device, dtype=torch.bool)

        for i in range(B):
            p, c = preds[i].unique_consecutive(return_counts=True)
            valid_mask_i = p != self.blank_idx
            x_splt = torch.split(x[i], c.tolist())
            out = torch.stack([t.mean(dim=0) for t in x_splt])
            if not valid_mask_i.any():
                # empty examples have just one blank
                x_compr[i, :1] = out
                lengths_compr[i] = 1
            else:
                out = out[valid_mask_i]
                x_compr[i, :out.size(0)] = out
                lengths_compr[i] = out.size(0)
                valid_examples[i] = True

        max_length = lengths_compr.max().item()

        x_compr = x_compr[:, :max_length]
        x_compr_mask = lengths_to_padding_mask(lengths_compr)

        if hasattr(self, "post"):
            x_compr = self.post(x_compr)
                
        if hasattr(self, "final_layernorm"):
            x_compr = self.final_layernorm(x_compr)
        
        decoder_out[-1]["compression_rate"] = (
            prev_lengths.float() - lengths_compr.float()
        ) / prev_lengths.float()
        speech_out["modified_valid_examples"] = [valid_examples]
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [lengths_compr]

        return decoder_out, speech_out