#!/usr/bin/env python3

import logging
from dataclasses import dataclass, field
from omegaconf import MISSING
from types import SimpleNamespace

import torch

from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from examples.extended_siamese.models.siamese_encoders_with_ctc import SiameseEncodersWithCTC

logger = logging.getLogger(__name__)


def dict_to_obj(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, (list, tuple)):
        return [dict_to_obj(x) for x in d]
    else:
        return d

@dataclass
class SiameseZeroShotS2TModelConfig(FairseqDataclass):
    encoder_path: str = field(
        default=MISSING,
        metadata={"help": "path to trained siamese encoder model"}
    )
    decoder_path: str = field(
        default=MISSING,
        metadata={"help": "path to trained MT model"}
    )
    not_load_submodules: bool = field(
        default=False,
        metadata={"help": "whether to load submodules"}
    )

@register_model("siamese_zs_s2t_model", dataclass=SiameseZeroShotS2TModelConfig)
class SiameseZeroShotS2TModel(BaseFairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def build_encoder(cfg, task):
        ckpt = torch.load(cfg.encoder_path)
        model_args = ckpt["cfg"]["model"]
        model_args["remove"] = True
        model_args = dict_to_obj(model_args)
        task.src_dict = task.tgt_dict
        encoder = SiameseEncodersWithCTC.build_model(model_args, task)
        if not cfg.not_load_submodules:
            logger.info(f"Loading encoder from {cfg.encoder_path} ...")
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt["model"], strict=False)
            if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
            if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        return encoder
    
    @staticmethod
    def build_decoder(cfg, task):
       
        ckpt = torch.load(cfg.decoder_path)
        model_args = ckpt["cfg"]["model"]
        
        dec_emb = Embedding(
            len(task.tgt_dict), model_args.decoder_embed_dim, task.tgt_dict.pad()
        )
        decoder = TransformerDecoder(model_args, task.tgt_dict, dec_emb)
        model_ckpt = {}
        for k, v in ckpt["model"].items():
            if k.startswith("decoder."):
                model_ckpt[k.replace("decoder.", "")] = v
        
        if not cfg.not_load_submodules:
            logger.info(f"Loading decoder from {cfg.decoder_path} ...")
            missing_keys, unexpected_keys = decoder.load_state_dict(model_ckpt, strict=False)
            if missing_keys: logger.info(f"Missing keys in state dict (some may correspond to resetted parameters):\n\t" + '\n\t'.join(missing_keys))
            if unexpected_keys: logger.info(f"Unexpected keys in state dict:\n\t" + '\n\t'.join(unexpected_keys))
        
        return decoder
    
    @classmethod
    def build_model(cls, cfg, task):
        cfg = SimpleNamespace(**cfg)
        encoder = cls.build_encoder(cfg, task)
        decoder = cls.build_decoder(cfg, task)
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