import logging
from dataclasses import dataclass, field
from omegaconf import II, MISSING

import torch.nn as nn


from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqDecoder
from fairseq.modules import FairseqDropout, LayerNorm

logger = logging.getLogger(__name__)


@dataclass
class CTCDecoderConfig(FairseqDataclass):
    embed_dim: int = field(
        default=II("model.embed_dim"),
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

class CTCDecoder(FairseqDecoder):
    def __init__(self, cfg: CTCDecoderConfig):
        
        try:
            dictionary = Dictionary.load(cfg.dictionary_path)
        except:
            dictionary = Dictionary.load('/home/usuaris/scratch/ioannis.tsiamas/pretrained_models/wav2vec2/dict.ltr.txt')
        # correct the ctc dictionary
        dictionary.symbols[0], dictionary.symbols[1] = dictionary.symbols[1], dictionary.symbols[0]
        dictionary.indices["<s>"], dictionary.indices["<pad>"] = 1, 0
        dictionary.bos_index, dictionary.pad_index = 1, 0
    
        super().__init__(dictionary)

        self.cfg = cfg
        self.dictionary = dictionary
        self.blank_idx = dictionary.pad()
        self.sep_token = "|"
        self.sep_idx = dictionary.symbols.index(self.sep_token)

        # only if the expected input is not the final output of the speech encoder
        if cfg.layernorm:
            self.layer_norm = LayerNorm(cfg.embed_dim)

        self.dropout_module = FairseqDropout(cfg.dropout)
        self.proj = nn.Linear(cfg.embed_dim, len(dictionary), bias=True)

        logger.info(f"| dictionary for CTC module: {len(dictionary)} types")

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