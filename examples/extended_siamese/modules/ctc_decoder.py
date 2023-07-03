import logging
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.models import FairseqDecoder
from fairseq.modules import FairseqDropout, LayerNorm

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.scale = embed_dim ** 0.5

    def forward(self, x, padding_mask=None):
        # x: [B, N, D], mask: [B, N]

        # Compute dot product between x and the learned query
        scores = torch.einsum('bnd,d->bn', x, self.query) / self.scale # [B, N]

        # Apply mask - set the scores to a large negative value where mask is True
        if padding_mask is not None:
            scores.masked_fill_(padding_mask, float('-inf'))

        # Apply softmax to get the attention weights
        attn_weights = F.softmax(scores, dim=-1) # [B, N]

        # Compute the weighted sum of x along the sequence dimension
        y = torch.einsum('bn,bnd->bd', attn_weights, x) # [B, D]

        return y
    
class LearnedScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(LearnedScaledDotProductAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x, padding_mask=None):
        # x: [B, N, D], mask: [B, N]

        # Compute query, key and value: apply learned linear transformations
        q = self.query(x)  # [B, N, D]
        k = self.key(x)    # [B, N, D]
        v = self.value(x)  # [B, N, D]
        
        # Compute dot product between query q and key k, and scale it
        scores = torch.einsum('bnd,bnd->bn', q, k) / self.scale  # [B, N]

        # Apply mask - set the scores to a large negative value where mask is True
        if padding_mask is not None:
            scores.masked_fill_(padding_mask, float('-inf'))

        # Apply softmax to get the attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, N]

        # Compute the weighted sum of values along the sequence dimension
        y = torch.einsum('bn,bnd->bd', attn_weights, v)  # [B, D]

        return y


@dataclass
class CTCBasedCompressionConfig(FairseqDataclass):
    type: ChoiceEnum(["letter", "word"]) = field(
        default="letter",
        metadata={"help": "ctc-based compression type"}
    )
    letter_pooling_fn: ChoiceEnum(["mean", "max", "attention", "learned_attention"]) = field(
        default="mean",
        metadata={"help": "pooling function for collapsing representations of consecutive chars"}
    )
    word_pooling_fn: ChoiceEnum(["mean", "max", "attention", "learned_attention"]) = field(
        default="mean",
        metadata={"help": "pooling function for collapsing char representations into word representations"}
    )
    transformer_layers: int = field(
        default=0,
        metadata={"help": "number of letter_transformer layers for word-level compression"}
    )
    adaptor: bool = field(
        default=False,
        metadata={"help": "whether to use an adaptor for word-level compression"}
    )
    projection_dim: int = field(
        default=8192,
        metadata={"help": "projection dimension for word-level compression"}
    )
    adaptor_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout rate for the adaptor"}
    )


@dataclass
class CTCDecoderConfig(FairseqDataclass):
    embed_dim: int = field(
        default=1024,
        metadata={"help": "embedding dimension"}
    )
    dictionary_path: str = field(
        default=MISSING,
        metadata={"help": "path to the ctc model dictionary for inference"}
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout rate before the ctc projection layer"}
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
        
        if self.cfg.ctc_compression is not None:
            
            if not hasattr(self.cfg.ctc_compression, "letter_pooling_fn"):
                self.cfg.ctc_compression.letter_pooling_fn = self.cfg.ctc_compression.pooling_fn
            
            if self.cfg.ctc_compression.letter_pooling_fn == "attention":
                self.letter_attention = ScaledDotProductAttention(cfg.embed_dim)
            elif self.cfg.ctc_compression.letter_pooling_fn == "learned_attention":
                self.letter_attention = LearnedScaledDotProductAttention(cfg.embed_dim)
            
            if self.cfg.ctc_compression.type == "word":
                
                if self.cfg.ctc_compression.word_pooling_fn == "attention":
                    self.word_attention = ScaledDotProductAttention(cfg.embed_dim)
                elif self.cfg.ctc_compression.word_pooling_fn == "learned_attention":
                    self.word_attention = LearnedScaledDotProductAttention(cfg.embed_dim)
                        
                
                if self.cfg.ctc_compression.transformer_layers > 0:
                    from examples.extended_siamese.modules import ContextEncoder, ContextEncoderConfig
                    transformer_cfg = ContextEncoderConfig()
                    transformer_cfg.encoder.embed_dim = cfg.embed_dim
                    transformer_cfg.encoder.ffn_embed_dim = cfg.embed_dim * 4
                    transformer_cfg.encoder.normalize_before = True
                    transformer_cfg.dropout = cfg.dropout
                    transformer_cfg.attention_dropout = cfg.dropout
                    transformer_cfg.activation_dropout = cfg.dropout
                    transformer_cfg.encoder.attention_heads = 16
                    transformer_cfg.encoder.layers = cfg.ctc_compression.transformer_layers
                    self.letter_transformer = ContextEncoder(transformer_cfg, no_final_layer_norm=True)
                    
                if self.cfg.ctc_compression.adaptor:
                    self.letter_adaptor = nn.Sequential(
                        LayerNorm(cfg.embed_dim),
                        nn.Linear(cfg.embed_dim, cfg.ctc_compression.projection_dim),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout),
                        nn.Linear(cfg.ctc_compression.projection_dim, cfg.embed_dim)
                    )
                    self.dropout = FairseqDropout(cfg.ctc_compression.adaptor_dropout)

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
        decoder_out, speech_out = self.letter_compression(decoder_out, speech_out)
        if self.cfg.ctc_compression.type == "word":
            decoder_out, speech_out = self.word_compression(decoder_out, speech_out)
        return decoder_out, speech_out

    def letter_pool(self, x):
        # TODO remove later
        # for backward compatibility
        if self.cfg.ctc_compression.letter_pooling_fn == "mean":
            y = torch.mean(x, dim=0)
        elif self.cfg.ctc_compression.letter_pooling_fn == "max":
            y = torch.max(x, dim=0)[0]
        elif self.cfg.ctc_compression.letter_pooling_fn in ["attention", "learned_attention"]:
            if x.size(0) > 1:
                y = self.letter_attention(x.unsqueeze(0))
                y = y.squeeze(0)
            else:
                y = x[0]
        return y
        
    def word_pool(self, x, mask=None, lens=None):
        if self.cfg.ctc_compression.word_pooling_fn == "mean":
            lens_ = lens.to(x.dtype)
            lens_[lens_ == .0] = 1.5 # to avoid division by 0
            y = torch.sum(x, dim=1) / lens_.unsqueeze(1)
        elif self.cfg.ctc_compression.word_pooling_fn == "max":
            y = torch.max(x, dim=1)[0]
        elif self.cfg.ctc_compression.word_pooling_fn in ["attention", "learned_attention"]:
            y = torch.zeros(x.size(0), x.size(-1), dtype=x.dtype, device=x.device)
            valid_mask = mask.eq(0).any(dim=1)
            y[valid_mask] = self.word_attention(x[valid_mask], mask[valid_mask])
        return y

    def letter_compression(self, decoder_out, speech_out):
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
        p_compr = torch.zeros(B, N, device=x.device, dtype=torch.long) # B x N
        lengths_compr = torch.zeros(B, device=x.device, dtype=torch.long) # B
        valid_examples = torch.zeros(B, device=x.device, dtype=torch.bool) # B

        for i in range(B):
            # p: compressed sequence, [N_i]
            # c: repeating counts for each j element of p, [N_i]
            p, c = preds[i].unique_consecutive(return_counts=True)
            valid_mask_i = p != self.blank_idx # [N_i]
            x_splt = torch.split(x[i], c.tolist()) # Tuple[tensor] of size N_i, x_splt[0] is the first c[0] elements of x[i]
            
            # TODO: this can be done in parallel
            # out = pad_sequence([x_split, batch_first=True, padding_value=0)
            # out = self.letter_pool(out, lens)
            out = torch.stack([self.letter_pool(t) for t in x_splt]) # N_i x D
            if not valid_mask_i.any():
                # empty examples have just one blank
                x_compr[i, :1] = out
                lengths_compr[i] = 1
            else:
                # TODO: after removing pads we can have consecutive letters that are not collapsed
                # remove blank tokens
                out = out[valid_mask_i] # N_i' x D
                p = p[valid_mask_i] # N_i'

                x_compr[i, :out.size(0)] = out
                p_compr[i, :out.size(0)] = p
                lengths_compr[i] = out.size(0)
                valid_examples[i] = True

        # real max length after removing blanks
        max_length = lengths_compr.max().item()

        x_compr = x_compr[:, :max_length]
        p_compr = p_compr[:, :max_length]
        x_compr_mask = lengths_to_padding_mask(lengths_compr)
        
        decoder_out[-1]["letter_compression_rate"] = prev_lengths.float() / lengths_compr.float()
        decoder_out[-1]["comression_rate"] = decoder_out[-1]["letter_compression_rate"]
        decoder_out[-1]["compressed_predictions"] = p_compr
        
        speech_out["modified_valid_examples"] = [valid_examples]
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [lengths_compr]

        return decoder_out, speech_out
    
    def word_compression(self, decoder_out, speech_out):       
        x = speech_out["modified_out"][0] # B x T x D
        preds = decoder_out[-1]["compressed_predictions"] # B x T
        lens = speech_out["modified_out_lengths"][0] # B
        
        sep_mask = preds.eq(self.sep_idx) # B x T

        B, T, D = x.size()
        dev = x.device

        word_lengths_list = []
        for i in range(B):
            
            # Find the indices where the separators are present in the sentence
            sep_indices = sep_mask[i].nonzero().squeeze(1)
            
            # account for missing last separator
            no_last_sep = preds[i, lens[i] - 1] != self.sep_idx
            if no_last_sep:
                sep_indices = torch.cat([sep_indices, torch.tensor([lens[i]], device=dev)])
            
            # account for consecutive separators
            diffs = torch.diff(sep_indices, append=sep_indices[-1].unsqueeze(0) + 2)
            transitions = diffs != 1
            if transitions.any():
                # Select only the first element of each set of consecutive numbers
                sep_indices = sep_indices[transitions]

            # Compute word lengths
            sep_indices[:lens[i]] += 1
            zero_tensor = torch.tensor([0], device=dev)
            word_lengths = torch.diff(sep_indices, prepend=zero_tensor)
            if no_last_sep:
                word_lengths[-1] -= 1
            
            word_lengths_list.append(word_lengths)
            
        word_lengths = pad_sequence(word_lengths_list, batch_first=True, padding_value=0)
        
        # create a tensor to fill-in
        num_words = word_lengths.size(1)
        max_word_length = word_lengths.max()
        x_words = torch.zeros(B, num_words, max_word_length, D, device=x.device, dtype=x.dtype)

        for i in range(B):
            x_words_i = torch.split(x[i, :lens[i]], word_lengths_list[i].tolist()) # Tuple[2d tensor]
            x_words_i = pad_sequence(x_words_i, batch_first=True, padding_value=0) # num_words_i x max_word_length_i x D
            x_words[i, :x_words_i.size(0), :x_words_i.size(1)] = x_words_i
            
        x_words = x_words.view(B * num_words, max_word_length, D)
        word_lengths = word_lengths.view(B * num_words)
        word_padding_mask = lengths_to_padding_mask(word_lengths)

        if hasattr(self, "letter_transformer"):
            x_words = x_words.transpose(0, 1)
            valid_mask = word_padding_mask.eq(0).any(dim=-1)
            x_words[:, valid_mask] = self.letter_transformer(x_words[:, valid_mask], padding_mask=word_padding_mask[valid_mask])[0]
            x_words = x_words.transpose(0, 1)
            x_words.masked_fill_(word_padding_mask.unsqueeze(-1), 0)
            
        if hasattr(self, "letter_adaptor"):
            x_words = x_words + self.dropout(self.letter_adaptor(x_words))
        
        x_compr = self.word_pool(x_words, lens=word_lengths, mask=word_padding_mask) # B * num_words x D
        
        x_compr = x_compr.view(B, num_words, D) # B x num_words x D
        x_compr_lens = (word_lengths.view(B, -1) != 0).sum(dim=-1) # B
        x_compr_mask = lengths_to_padding_mask(x_compr_lens) # B x num_words
        x_compr.masked_fill_(x_compr_mask.unsqueeze(-1), 0)

        decoder_out[-1]["word_compression_rate"] = lens.float() / x_compr_lens.float()
        decoder_out[-1]["compression_rate"] = decoder_out[-1]["word_compression_rate"] * decoder_out[-1]["letter_compression_rate"]
        
        speech_out["modified_out"] = [x_compr]
        speech_out["modified_padding_mask"] = [x_compr_mask]
        speech_out["modified_out_lengths"] = [x_compr_lens]
        
        return decoder_out, speech_out