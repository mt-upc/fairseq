import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqDecoder
from fairseq.modules import FairseqDropout, LayerNorm


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    logging.info(f"| bias in Linear layer: {bias}")
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
    pooling_fn: str = field(
        default="mean", metadata={"help": "pooling function: max, mean, attention"}
    )
    layernorm: bool = field(
        default=False,
        metadata={
            "help": "whether to use layer normalization before ctc projection layer"
        },
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

        logging.info(f"| dictionary for CTC module: {len(dictionary)} types")
        logging.info(f"| CTC-based compression: {cfg.ctc_compression}")
        logging.info(f"| CTC-based compression type: {cfg.ctc_compression_type}")

        if cfg.ctc_compression and cfg.post_compression_layer:
            self.post = nn.Sequential(
                LayerNorm(cfg.embed_dim),
                nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
                nn.GELU(),
                nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            )
            logging.info(f"| post-compression layer: True")

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
            # TODO
            raise NotImplementedError

    def compress_word(self, decoder_out, speech_out):
        x = speech_out["encoder_out"][0].transpose(0, 1)  # T x B x D
        prev_lengths = speech_out["encoder_out_lengths"][0]

        with torch.no_grad():
            lprobs_ctc = F.log_softmax(decoder_out[0], dim=-1)  # T x B x V
            preds = torch.argmax(lprobs_ctc, dim=-1)  # T x B

        T, B, D = x.size()
        x = x.transpose(0, 1)  # B x T x D
        preds = preds.transpose(0, 1)  # B x T

        separetor_mask = preds == self.sep_idx
        valid_mask = preds != self.blank_idx

        t = valid_mask.sum(dim=-1).max().item()
        x_compr = torch.zeros(B, t, D, device=x.device, dtype=x.dtype)
        # preds_compr = []

        for i in range(B):
            # p = []
            if valid_mask[i].sum() == 0:
                continue
            x_i = x[i, valid_mask[i]]  # TODO chec if underlying memory is the same
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
                        # p.append(preds[i, prev_idx+1:idx].tolist())
                    prev_idx = idx
            # preds_compr.append(p)

        lengths_compr = (x_compr.sum(dim=-1) != self.blank_idx).sum(dim=-1)  # B

        max_length = lengths_compr.max().item()
        if not max_length:
            return decoder_out, speech_out

        x_compr = x_compr[:, :max_length]
        x_compr_mask = lengths_to_padding_mask(lengths_compr)

        if hasattr(self, "post"):
            x_compr = self.post(x_compr)

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

        T, B, D = x.size()
        x = x.transpose(0, 1)
        preds = preds.transpose(0, 1)

        x_compr = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        preds_after_merged = torch.zeros(B, T, dtype=torch.long, device=preds.device)

        # breakpoint()

        for i in range(B):
            p, c = preds[i].unique_consecutive(return_counts=True)

            x_splt = torch.split(x[i], c.tolist())
            out = torch.stack([t.mean(dim=0) for t in x_splt])

            x_compr[i, : out.size(0)] = out
            preds_after_merged[i, : p.size(0)] = p

        mask = preds_after_merged == self.blank_idx
        preds_after_merged.masked_fill_(mask, 0)
        x_compr.masked_fill_(mask.unsqueeze(-1), 0)

        # breakpoint()

        # Get mask of elements which are blank
        non_blank_mask = ~preds_after_merged.eq(self.blank_idx)
        # Get new lengths
        lengths = torch.sum(non_blank_mask, dim=-1)

        x_compr = x_compr.masked_select(non_blank_mask.unsqueeze(-1)).view(-1, D)
        x_compr = self._pad_seq_given_lens_arrays(x_compr, lengths)

        # breakpoint()

        if not x_compr.size(1):
            logging.info(
                "No frames left after CTC compression. Returning original output."
            )
            return decoder_out, speech_out

        x_compr_mask = (
            torch.arange(x_compr.size(1), device=x_compr.device).expand(
                x_compr.size(0), -1
            )
            >= lengths.unsqueeze(1)
        ).bool()

        decoder_out[-1]["compression_rate"] = (
            prev_lengths.float() - lengths.float()
        ) / prev_lengths.float()

        if hasattr(self, "post"):
            x_compr = self.post(x_compr)

        assert x_compr.size(0) == speech_out["encoder_out"][0].size(0)
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [lengths]

        return decoder_out, speech_out

    def _pad_seq_given_lens_arrays(self, input, lengths, padding_value=0.0):
        max_len = max(lengths)
        B = len(lengths)
        D = input.size(-1)
        padded = torch.full(
            (B, max_len, D), padding_value, device=input.device, dtype=input.dtype
        )
        cum_len = 0
        for i, val in enumerate(lengths):
            padded[i, :val] = input[cum_len : cum_len + val]
            cum_len += val
        return padded
