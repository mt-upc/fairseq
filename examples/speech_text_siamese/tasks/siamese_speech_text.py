# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
import sys
sys.path.append(os.environ['FAIRSEQ_ROOT'])

from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    SpeechToTextJointDatasetCreator
)
from fairseq.tasks import register_task
from examples.speech_text_joint_to_text.tasks.speech_text_joint import SpeechTextJointToTextTask


logger = logging.getLogger(__name__)


@register_task("siamese_speech_text_to_text")
class SiameseSpeechTextToTextTask(SpeechTextJointToTextTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--speech-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        parser.add_argument(
            "--text-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq  > 1",
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens per sample for monolingual dataset",
        )
        parser.add_argument(
            "--load-speech-only",
            action="store_true",
            help="load speech data only",
        )
        parser.add_argument(
            "--monolingual-text-data",
            default="",
            help="path to monolingual text data directory",
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            default=512,
            metavar="N",
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--max-positions-text",
            type=int,
            metavar="N",
            default=400,
            help="maximum tokens for per encoder text input ",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        self.speech_only = getattr(args, "load_speech_only", False)
        self.mask_idx = None
        self.mask_sym = "<mask>"
        self.args = args
        if self.src_dict is not None:
            assert self.tgt_dict.pad() == self.src_dict.pad()
            assert self.tgt_dict.eos() == self.src_dict.eos()
            if self.args.monolingual_text_data != "":
                self.mask_idx = self.src_dict.add_symbol(self.mask_sym)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if not os.path.isfile(src_dict_path):
            logging.warning("Dict not found: {}".format(src_dict_path))
        if not os.path.isfile(tgt_dict_path):
            raise FileNotFoundError("Dict not found: {}".format(tgt_dict_path))
        src_dict = Dictionary.load(src_dict_path.as_posix()) if src_dict_path.exists() else None
        if src_dict is not None:
            logger.info(
                f"source dictionary size ({data_cfg.src_vocab_filename}): " f"{len(src_dict):,}")
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}")

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        s2t_dataset = SpeechToTextJointDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            append_eos=True,
            use_src_lang_id=self.data_cfg.prepend_src_lang_tag
        )
        self.datasets[split] = s2t_dataset

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.src_dict