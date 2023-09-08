# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

from examples.speech_text_joint_to_text.tasks.speech_text_joint import (
    SpeechTextJointToTextTask,
    SpeechTextJointToTextTaskConfig,
)
from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
    SpeechToTextJointDatasetCreator,
)
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task("siamese_speech_text", dataclass=SpeechTextJointToTextTaskConfig)
class SiameseSpeechTextTask(SpeechTextJointToTextTask):
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TJointDataConfig(Path(cfg.data) / cfg.config_yaml)
        self.cfg = cfg
        
        # gather here some other task parameters from other configs to pass it to the dataset constructor
        self.data_cfg.config["ot_aux_layers"] = []
        if hasattr(self.cfg, "ot_aux_layers") and self.cfg.ot_aux_layers:
            self.data_cfg.config["ot_aux_layers"] = [int(l) for l in self.cfg.ot_aux_layers.split(",")]
        if hasattr(self.cfg, "mt_model_path") and self.cfg.mt_model_path:
            if "ckpts" in self.cfg.mt_model_path:
                # fine-tuned version
                self.data_cfg.config["mt_model_name"] = Path(self.cfg.mt_model_path).parent.parent.name
            else:
                # original version
                self.data_cfg.config["mt_model_name"] = Path(self.cfg.mt_model_path).stem
        if hasattr(self.cfg, "mt_num_layers") and self.cfg.mt_num_layers:
            self.data_cfg.config["mt_num_layers"]  = self.cfg.mt_num_layers

    @classmethod
    def setup_task(cls, cfg, **kwcfg):
        data_cfg = S2TJointDataConfig(Path(cfg.data) / cfg.config_yaml)
        tgt_dict_path = Path(cfg.data) / data_cfg.vocab_filename
        src_dict_path = Path(cfg.data) / data_cfg.src_vocab_filename
        if not os.path.isfile(src_dict_path):
            logging.warning("Dict not found: {}".format(src_dict_path))
        if not os.path.isfile(tgt_dict_path):
            raise FileNotFoundError("Dict not found: {}".format(tgt_dict_path))
        src_dict = (
            Dictionary.load(src_dict_path.as_posix())
            if src_dict_path.exists()
            else None
        )
        if src_dict is not None:
            logger.info(
                f"source dictionary size ({data_cfg.src_vocab_filename}): "
                f"{len(src_dict):,}"
            )
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())

        # correct the ctc dictionary
        tgt_dict.symbols[0], tgt_dict.symbols[1] = tgt_dict.symbols[1], tgt_dict.symbols[0],
        tgt_dict.indices["<s>"], tgt_dict.indices["<pad>"] = 1, 0
        tgt_dict.bos_index, tgt_dict.pad_index = 1, 0

        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwcfg):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.cfg)
        bpe_tokenizer = self.build_bpe(self.cfg)
        src_pre_tokenizer = self.build_src_tokenizer(self.cfg)
        src_bpe_tokenizer = self.build_src_bpe(self.cfg)
        s2t_dataset = SpeechToTextJointDatasetCreator.from_tsv(
            self.cfg.data,
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
            seed=self.cfg.seed,
            append_eos=True,
            use_src_lang_id=self.data_cfg.prepend_src_lang_tag,
        )
        self.datasets[split] = s2t_dataset

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict
