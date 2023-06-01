# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from argparse import Namespace
from omegaconf import II, MISSING
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from fairseq import utils, metrics, scoring
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform,
)
from fairseq.data.audio.speech_distillation_dataset import SpeechDistillationDataset

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.scoring.wer import WerScorerConfig
from fairseq.scoring.bleu import SacrebleuConfig
from fairseq.tasks import FairseqTask, register_task
from fairseq.utils import safe_hasattr

EVAL_BLEU_ORDER=4

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeDistillationConfig(FairseqDataclass):
    path: str = field(
        default="",
        metadata={"help": "path to teacher outputs: ${path}/${split}/*.pt"}
    )

@dataclass
class SpeechToTextTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "manifest root path"}
    )
    data_config_yaml: str = field(
        default="config.yaml",
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    max_source_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"}
    )

    # Reporting metrics during training
    eval_wer: bool = field(
        default=False,
        metadata={"help": "compute WER on the validation set"}
    )
    eval_wer_config: WerScorerConfig = field(
        default_factory=lambda: WerScorerConfig("wer"),
        metadata={"help": "WER scoring configuration"},
    )
    eval_bleu: bool = field(
        default=False,
        metadata={"help": "compute SacreBLEU on the validation set"}
    )
    eval_bleu_config: SacrebleuConfig = field(
        default_factory=lambda: SacrebleuConfig("sacrebleu"),
        metadata={"help": "SacreBLEU scoring configuration"},
    )
    eval_gen_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "generaton config for evaluating during training"},
    )
    eval_print_samples: bool = field(
        default=False,
        metadata={"help": "print sample generations during validation"}
    )

    # Inherit from other configs
    train_subset: str = II("dataset.train_subset")
    seed: int = II("common.seed")
    knowledge_distillation: Optional[KnowledgeDistillationConfig] = field(
        default=None,
        metadata={"help": "Knowledge distillation arguments"}
    )
    sampling_ratios: str = field(
        default="1",
        metadata={"help": "sampling ratios of the train subsets"}
    )
    interactive_tgt_lang: Optional[str] = field(
        default=None,
        metadata={"help": "Target language to be used with Fairseq's interactive mode."}
    )
    only_construct_model: bool = field(
        default=False,
        metadata={"help": "Only construct the model, save it, and exit."},
    )


@register_task("speech_to_text", dataclass=SpeechToTextTaskConfig)
class SpeechToTextTask(FairseqTask):

    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        try:
            self.data_cfg = S2TDataConfig(Path(cfg.data) / cfg.data_config_yaml)
        except AttributeError:
            # compatibility with siamese pre-training
            self.data_cfg = S2TDataConfig(Path(cfg.data) / cfg.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        self.pre_tokenizer = self.build_tokenizer(cfg)
        self.bpe_tokenizer = self.build_bpe(cfg)
        self.scorers = []
        
        self.eval_wer = getattr(cfg, "eval_wer", False)
        self.eval_bleu = getattr(cfg, "eval_bleu", False)
        
        if self.eval_wer:
            self.scorers.append(
                scoring.build_scorer(cfg.eval_wer_config, self.tgt_dict)
            )
        if self.eval_bleu:
            self.scorers.append(
                scoring.build_scorer(cfg.eval_bleu_config, self.tgt_dict)
            )
            
        self.sampling_ratios = None
        if getattr(cfg, "sampling_ratios", None) is not None:
            self.sampling_ratios = list(map(float, cfg.sampling_ratios.split(",")))
        
        self.use_kd = getattr(cfg, "knowledge_distillation", None) is not None and cfg.knowledge_distillation != ""

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.cfg.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        data_cfg = S2TDataConfig(Path(cfg.data) / cfg.data_config_yaml)
        dict_path = Path(data_cfg.vocab_filename)
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        return cls(cfg, tgt_dict)

    def build_criterion(self, cfg):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and cfg.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "criterion.ignore_prefix_size=1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(cfg, self)

    def load_dataset(self, split, epoch=1, **kwargs):
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.cfg.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.pre_tokenizer,
            self.bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
            speaker_to_id=self.speaker_to_id,
            sampling_ratios=self.sampling_ratios,
        )
        
        if is_train_split and self.use_kd:
            self.datasets[split] = SpeechDistillationDataset(
                self.datasets[split],
                self.cfg.knowledge_distillation.path,
                pad_idx=self.tgt_dict.pad()
            )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_model(self, cfg, from_checkpoint=False):
        model = super(SpeechToTextTask, self).build_model(cfg, from_checkpoint)
        
        if self.eval_bleu or self.eval_wer:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_gen_config
            )
            # Trick: update model configuration globally
            if not safe_hasattr(cfg.encoder, 'pre_args'):
                cfg.encoder.pre_args = model.encoder.cfg_.pre_args
            if not safe_hasattr(cfg.decoder, 'pre_args'):
                cfg.decoder.pre_args = model.decoder.cfg_.pre_args

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        def decode(toks):
            if hasattr(self.sequence_generator, "symbols_to_strip_from_output"):
                to_ignore = self.sequence_generator.symbols_to_strip_from_output
            else:
                to_ignore = {self.sequence_generator.eos}

            s = self.tgt_dict.string(
                toks.int().cpu(),
                escape_unk=True,
                extra_symbols_to_ignore=to_ignore
            )
            if self.bpe_tokenizer:
                s = self.bpe_tokenizer.decode(s)
            if self.pre_tokenizer:
                s = self.pre_tokenizer.decode(s)
            return s

        if len(self.scorers) > 0:
            prefix_tokens = sample['target'][:, 0].unsqueeze(1) if self.data_cfg.prepend_tgt_lang_tag else None
            gen_out = self.inference_step(self.sequence_generator, [model], sample, prefix_tokens=prefix_tokens)
            for i in range(len(gen_out)):
                ref_tok = utils.strip_pad(sample["target"][i], self.tgt_dict.pad()).int().cpu()
                pred_tok = gen_out[i][0]["tokens"].int().cpu()
                if self.data_cfg.prepend_tgt_lang_tag:
                    ref_tok = ref_tok[1:]
                    pred_tok = pred_tok[1:]
                ref = decode(ref_tok)
                pred = decode(pred_tok)
                for s in self.scorers:
                    s.add_string(ref, pred)

            if self.cfg.eval_print_samples:
                logger.info("Validation example:")
                logger.info("H-{} {}".format(sample["id"][-1], pred))
                logger.info("T-{} {}".format(sample["id"][-1], ref))

        for s in self.scorers:
            if s.cfg._name == 'wer':
                logging_output["_wer_distance"] = s.distance
                logging_output["_wer_ref_len"] = s.ref_length
            elif s.cfg._name == 'sacrebleu':
                sacrebleu_out = s._score()
                logging_output["_bleu_sys_len"] = sacrebleu_out.sys_len
                logging_output["_bleu_ref_len"] = sacrebleu_out.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(sacrebleu_out.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = sacrebleu_out.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = sacrebleu_out.totals[i]
            else:
                raise NotImplemented()

            if safe_hasattr(s, "reset"):
                s.reset()
            else:
                s.ref = []
                s.pred = []

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        for s in self.scorers:
            if s.cfg._name == 'wer':
                if  sum_logs("_wer_ref_len") > 0:
                    metrics.log_scalar("_wer_distance", sum_logs("_wer_distance"))
                    metrics.log_scalar("_wer_ref_len", sum_logs("_wer_ref_len"))

                    def compute_wer(meters):
                        import torch
                        ref_len = meters["_wer_ref_len"].sum
                        wer = meters["_wer_distance"].sum / ref_len
                        if torch.is_tensor(wer):
                            wer = wer.cpu().item()
                        return round(100 * wer, 2)

                    metrics.log_derived("wer", compute_wer)

            elif s.cfg._name == 'sacrebleu':
                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(sum_logs("_bleu_counts_" + str(i)))
                    totals.append(sum_logs("_bleu_totals_" + str(i)))

                if max(totals) > 0:
                    # log counts as numpy arrays -- log_scalar will sum them correctly
                    metrics.log_scalar("_bleu_counts", np.array(counts))
                    metrics.log_scalar("_bleu_totals", np.array(totals))
                    metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                    metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                    def compute_bleu(meters):
                        import inspect
                        import torch

                        try:
                            from sacrebleu.metrics import BLEU

                            comp_bleu = BLEU.compute_bleu
                        except ImportError:
                            # compatibility API for sacrebleu 1.x
                            import sacrebleu

                            comp_bleu = sacrebleu.compute_bleu

                        fn_sig = inspect.getfullargspec(comp_bleu)[0]
                        if "smooth_method" in fn_sig:
                            smooth = {"smooth_method": "exp"}
                        else:
                            smooth = {"smooth": "exp"}
                        bleu = comp_bleu(
                            correct=meters["_bleu_counts"].sum,
                            total=meters["_bleu_totals"].sum,
                            sys_len=meters["_bleu_sys_len"].sum if torch.is_tensor(meters["_bleu_sys_len"].sum) == False else meters["_bleu_sys_len"].sum.long().item(),
                            ref_len=meters["_bleu_ref_len"].sum if torch.is_tensor(meters["_bleu_ref_len"].sum) == False else meters["_bleu_ref_len"].sum.long().item(),
                            **smooth,
                        )
                        return round(bleu.score, 2)

                    metrics.log_derived("sacrebleu", compute_bleu)

            else:
                raise NotImplemented()

    def build_generator(
        self,
        models,
        cfg,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and cfg.prefix_size != 1:
            raise ValueError(
                'Please set "generation.prefix_size=1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
        return super().build_generator(
            models, cfg, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, cfg):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, cfg):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
