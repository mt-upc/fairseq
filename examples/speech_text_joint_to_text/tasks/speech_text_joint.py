# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import II

import torch
from fairseq.data import (
    encoders,
    Dictionary,
    ResamplingDataset,
    TransformEosLangPairDataset,
    ConcatDataset,
)
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    LangPairMaskDataset,
    ModalityDatasetItem,
)
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
    SpeechToTextJointDatasetCreator,
)
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask, SpeechToTextTaskConfig
from fairseq.tasks.translation import load_langpair_dataset

logger = logging.getLogger(__name__)
LANG_TAG_TEMPLATE = "<lang:{}>"

@dataclass
class SpeechTextJointToTextTaskConfig(SpeechToTextTaskConfig):
    parallel_text_data: str = field(
        default="",
        metadata={"help": "path to parallel text data directory"},
    )
    max_tokens_text: Optional[int] = field(
        default=None,
        metadata={"help": "maximum tokens for encoder text input"},
    )
    max_positions_text: int = field(
        default=400,
        metadata={"help": "maximum tokens for per encoder text input"},
    )
    langpairs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": 'language pairs for text training, separated with ","'
        },
    )
    speech_sample_ratio: float = field(
        default=1,
        metadata={
            "help": "Multiple Ratio for speech dataset with transcripts"
        },
    )
    text_sample_ratio: float = field(
        default=1,
        metadata={"help": "Multiple Ratio for text set"},
    )
    update_mix_data: bool = field(
        default=False,
        metadata={
            "help": "use mixed data in one update when update-freq > 1"
        },
    )
    load_speech_only: bool = field(
        default=False,
        metadata={"help": "load speech data only"},
    )
    mask_text_ratio: float = field(
        default=0.0,
        metadata={"help": "mask V source tokens for text only mode"},
    )
    mask_text_type: str = field(
        default="random",
        metadata={
            "help": "mask text typed",
            "choices": ["random", "tail"]
        },
    )
    noise_token: str = field(
        default="",
        metadata={
            "help": "noise token for masking src text tokens if mask-text-ratio > 0"
        },
    )
    infer_target_lang: str = field(
        default="",
        metadata={"help": "target language for inference"},
    )
    ot_aux_layers: str = field(
        default=II("criterion.ot_teacher_aux_layers"),
        metadata={"help": "auxiliary layers for OT loss"},
    )
    mt_model_path: str = field(
        default=II("model.text_encoder.path"),
        metadata={"help": "path to the mt model"
                  "useful to get the name of the model and pass it to the dataset"
                  "in order to load the cached representations from that encoder"}
    )
    mt_num_layers: int = field(
        default=II("model.text_encoder.num_layers"),
        metadata={"help": "number of layers in the MT encoder"}
    )


@register_task("speech_text_joint_to_text", dataclass=SpeechTextJointToTextTaskConfig)
class SpeechTextJointToTextTask(SpeechToTextTask):
    """
    Task for joint training speech and text to text.
    """

    def __init__(self, cfg, src_dict, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(cfg, tgt_dict)
        self.src_dict = src_dict
        self.data_cfg = S2TJointDataConfig(Path(cfg.data) / cfg.config_yaml)
        self.speech_only = cfg.load_speech_only
        self._infer_tgt_lang_id = infer_tgt_lang_id

    @classmethod
    def setup_task(cls, cfg, **kwcfg):
        """Setup the task (e.g., load dictionaries)."""
        data_cfg = S2TJointDataConfig(Path(cfg.data) / cfg.config_yaml)
        tgt_dict_path = Path(cfg.data) / data_cfg.vocab_filename
        src_dict_path = Path(cfg.data) / data_cfg.src_vocab_filename
        if (not os.path.isfile(src_dict_path)) or (not os.path.isfile(tgt_dict_path)):
            raise FileNotFoundError("Dict not found: {}".format(cfg.data))
        src_dict = Dictionary.load(src_dict_path.as_posix())
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())

        print("| src dictionary: {} types".format(len(src_dict)))
        print("| tgt dictionary: {} types".format(len(tgt_dict)))

        if cfg.parallel_text_data != "":
            if not os.path.isabs(cfg.parallel_text_data):
                cfg.parallel_text_data = os.path.join(
                    cfg.data, cfg.parallel_text_data
                )

            if cfg.langpairs is None:
                raise Exception(
                    "Could not infer language pair, please provide it explicitly"
                )
        infer_tgt_lang_id = None
        if cfg.infer_target_lang != "" and data_cfg.prepend_tgt_lang_tag_no_change:
            tgt_lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(
                cfg.infer_target_lang
            )
            infer_tgt_lang_id = tgt_dict.index(tgt_lang_tag)
            assert infer_tgt_lang_id != tgt_dict.unk()
        return cls(cfg, src_dict, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)

    def load_langpair_dataset(
        self, prepend_tgt_lang_tag=False, sampling_alpha=1.0, epoch=0
    ):
        lang_pairs = []
        text_dataset = None
        split = "train"
        for lp in self.cfg.langpairs.split(","):
            src, tgt = lp.split("-")
            text_dataset = load_langpair_dataset(
                self.cfg.parallel_text_data,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.cfg.max_positions_text,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=False,
                truncate_source=False,
            )
            if prepend_tgt_lang_tag:
                # TODO
                text_dataset = TransformEosLangPairDataset(
                    text_dataset,
                    src_eos=self.src_dict.eos(),
                    tgt_bos=self.tgt_dict.eos(),  # 'prev_output_tokens' starts with eos
                    new_tgt_bos=self.tgt_dict.index(LANG_TAG_TEMPLATE.format(tgt)),
                )
            lang_pairs.append(text_dataset)
        if len(lang_pairs) > 1:
            if sampling_alpha != 1.0:
                size_ratios = SpeechToTextDatasetCreator.get_size_ratios(
                    self.cfg.langpairs.split(","),
                    [len(s) for s in lang_pairs],
                    alpha=sampling_alpha,
                )
                lang_pairs = [
                    ResamplingDataset(d, size_ratio=r, epoch=epoch, replace=(r >= 1.0))
                    for d, r in zip(lang_pairs, size_ratios)
                ]
            return ConcatDataset(lang_pairs)
        return text_dataset

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=self._infer_tgt_lang_id,
            )

    def build_src_tokenizer(self, cfg):
        logger.info(f"src-pre-tokenizer: {self.data_cfg.src_pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.src_pre_tokenizer))

    def build_src_bpe(self, cfg):
        logger.info(f"tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def load_dataset(self, split, epoch=1, combine=False, **kwcfg):
        """Load a given dataset split.

        cfg:
            split (str): name of the split (e.g., train, valid, test)
        """
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.cfg)
        bpe_tokenizer = self.build_bpe(self.cfg)
        src_pre_tokenizer = self.build_src_tokenizer(self.cfg)
        src_bpe_tokenizer = self.build_src_bpe(self.cfg)
        ast_dataset = SpeechToTextJointDatasetCreator.from_tsv(
            self.cfg.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=None if self.speech_only else self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
        )
        noise_token_id = -1
        text_dataset = None
        if self.cfg.parallel_text_data != "" and is_train_split:
            text_dataset = self.load_langpair_dataset(
                self.data_cfg.prepend_tgt_lang_tag_no_change, 1.0, epoch=epoch,
            )
            if self.cfg.mask_text_ratio > 0:
                # add mask
                noise_token_id = (
                    self.src_dict.unk()
                    if self.cfg.noise_token == ""
                    else self.src_dict.index(self.cfg.noise_token)
                )
                text_dataset = LangPairMaskDataset(
                    text_dataset,
                    src_bos=self.src_dict.bos(),
                    src_eos=self.src_dict.eos(),
                    noise_id=noise_token_id,
                    mask_ratio=self.cfg.mask_text_ratio,
                    mask_type=self.cfg.mask_text_type,
                )

        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "sup_speech",
                    ast_dataset,
                    (self.cfg.max_source_positions, self.cfg.max_target_positions),
                    self.cfg.max_tokens,
                    self.cfg.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.cfg.max_positions_text, self.cfg.max_target_positions),
                    self.cfg.max_tokens_text
                    if self.cfg.max_tokens_text is not None
                    else self.cfg.max_tokens,
                    self.cfg.batch_size,
                ),
            ]
            ast_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = ast_dataset

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None if self.speech_only else self.src_dict

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):

        if not isinstance(dataset, MultiModalityDataset):
            return super(SpeechTextJointToTextTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )

        mult_ratio = [self.cfg.speech_sample_ratio, self.cfg.text_sample_ratio]
        assert len(dataset.datasets) == 2

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1 if self.cfg.update_mix_data else max(self.cfg.update_freq),
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter
