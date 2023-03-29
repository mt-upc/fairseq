# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
import editdistance

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from fairseq.data.data_utils import post_process
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.logging.meters import safe_round
from fairseq.tasks import FairseqTask

SAMPLE_LOSS_CHOICES = ChoiceEnum(
    ["sinkhorn", "hausdorff", "energy", "gaussian", "laplacian"]
)


@dataclass
class CtcWassersteinCriterionConfig(CtcCriterionConfig):
    ctc_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for CTC loss"},
    )
    ot_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for OT loss"},
    )
    norm_before_ot: bool = field(
        default=False,
        metadata={"help": "Normalize before computing OT"},
    )
    ot_loss: SAMPLE_LOSS_CHOICES = field(
        default="sinkhorn",
        metadata={"help": "type of distance measure between X_i and Y_j"},
    )
    ot_p: int = field(
        default=2,
        metadata={"help": "p in SampleLoss"},
    )
    ot_blur: float = field(
        default=0.05,
        metadata={"help": "blur in SampleLoss"},
    )
    ot_scaling: float = field(
        default=0.5,
        metadata={"help": "scaling in SampleLoss"},
    )
    ot_positional_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for OT positional embeddings"},
    )
    detach_text_encoder: bool = field(
        default=False,
        metadata={"help": "Detach text encoder from the graph"},
    ) # TODO: currently there is also freeze-text-enc. unify them


@register_criterion(
    "ctc_wass_loss", dataclass=CtcWassersteinCriterionConfig
)
class CtcWassersteinCriterion(CtcCriterion):
    def __init__(self, cfg: CtcWassersteinCriterionConfig, task: FairseqTask):
        super().__init__(cfg, task)

        self.ctc_weight = cfg.ctc_weight
        self.ot_weight = cfg.ot_weight

        self.detach_text_encoder = cfg.detach_text_encoder
        self.norm_before_ot = cfg.norm_before_ot
        self.ot_loss = cfg.ot_loss
        self.ot_p = cfg.ot_p
        self.ot_blur = cfg.ot_blur
        self.ot_scaling = cfg.ot_scaling
        self.ot_positional_weight = cfg.ot_positional_weight

        logging.info(f"*** Loss function ***")
        logging.info(f"ctc_weight = {self.ctc_weight}")
        logging.info(f"ot_weight = {self.ot_weight}")
        logging.info(
            f"ot_loss = {self.ot_loss}, ot_p = {self.ot_p}, ot_blur = {self.ot_blur}, ot_scaling = {self.ot_scaling}"
        )

        self.ot_loss = SamplesLoss(
            loss=cfg.ot_loss, p=self.ot_p, blur=self.ot_blur, scaling=self.ot_scaling
        )

    def forward(self, model, sample):
        net_input = sample["net_input"]
        net_output, encoder_out = model(
            src_tokens=net_input["src_tokens"],
            src_lengths=net_input["src_lengths"],
            src_txt_tokens=net_input["src_txt_tokens"],
            src_txt_lengths=net_input["src_txt_lengths"],
        )
        sample_size = (
            net_input["src_tokens"].size(0)
            if self.sentence_avg
            else sample["ntokens"]
        )

        loss = 0.0
        extra = {"ctc_loss": 0.0, "wass_loss": 0.0}

        if self.ctc_weight > 0.0:
            ctc_loss, extra = self.compute_ctc_loss(
                model, net_output, encoder_out, sample["target"], extra
            )
            loss += self.ctc_weight * ctc_loss

        if self.ot_weight > 0.0:
            wass_loss, num_nans = self.compute_wass_loss(self.ot_loss, encoder_out)
            loss += self.ot_weight * wass_loss
            extra["wass_loss"] = wass_loss
            extra["num_nans"] = num_nans

        logging_output = {
            "loss": utils.item(loss.data)
            if loss != 0.0
            else 0.0,  # * sample['ntokens'],
            "ctc_loss": utils.item(extra["ctc_loss"].data)
            if extra["ctc_loss"] != 0.0
            else 0.0,
            "wass_loss": utils.item(extra["wass_loss"].data)
            if extra["wass_loss"] != 0.0
            else 0.0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["id"].numel(),
            "sample_size": net_input["src_tokens"].size(0)
            if self.sentence_avg
            else sample["ntokens"],
            "num_nans": utils.item(extra["num_nans"].data),
        }
        
        if "reduced_speech_output" in net_output[-1]:
            logging_output["reduced_speech_output"] = utils.item(
                net_output[-1]["reduced_speech_output"].data
            )

        if not model.training and self.ctc_weight > 0.0:
            logging_output = self.compute_wer(
                extra["lprobs_ctc"],
                sample,
                extra["input_lengths"],
                logging_output,
            )

        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, net_output, encoder_out, targets, extra):
        lprobs = model.get_normalized_probs(
            net_output,
            log_probs=True,
            idx=1,
        ).contiguous()  # (T, B, C) from the encoder

        spch_encoder_out = (
            encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
        )
        if spch_encoder_out["encoder_padding_mask"][0] is not None:
            non_padding_mask = ~spch_encoder_out["encoder_padding_mask"][0]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )
        pad_mask = (targets != self.pad_idx) & (targets != self.eos_idx)
        targets_flat = targets.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        extra["ctc_loss"] = ctc_loss
        extra["lprobs_ctc"] = lprobs
        extra["input_lengths"] = input_lengths

        return ctc_loss, extra
    
    def compute_wer(self, lprobs, sample, input_lengths, logging_output):

        with torch.no_grad():
            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0
            wv_errs = 0
            for lp, t, inp_l in zip(lprobs_t, sample["target"], input_lengths,
            ):
                lp = lp[:inp_l].unsqueeze(0)
                decoded = None
                if self.w2l_decoder is not None:
                    decoded = self.w2l_decoder.decode(lp)
                    if len(decoded) < 1:
                        decoded = None
                    else:
                        decoded = decoded[0]
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]

                p = (t != self.task.target_dictionary.pad()) & (
                    t != self.task.target_dictionary.eos()
                )
                targ = t[p]
                targ_units = self.task.target_dictionary.string(targ)
                targ_units_arr = targ.tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

                targ_words = post_process(targ_units, self.post_process).split()

                pred_units = self.task.target_dictionary.string(pred_units_arr)
                pred_words_raw = post_process(pred_units, self.post_process).split()

                if decoded is not None and "words" in decoded:
                    pred_words = decoded["words"]
                    w_errs += editdistance.eval(pred_words, targ_words)
                    wv_errs += editdistance.eval(pred_words_raw, targ_words)
                else:
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                w_len += len(targ_words)
            logging_output["wv_errors"] = wv_errs
            logging_output["w_errors"] = w_errs
            logging_output["w_total"] = w_len
            logging_output["c_errors"] = c_err
            logging_output["c_total"] = c_len
        return logging_output

    def compute_wass_loss(self, ot_loss, encoder_out):
        
        if "adaptor_out" in encoder_out[0]:
            key = "adaptor"
        elif "compressed_out" in encoder_out[0]:
            key = "compressed"
        else:
            key = "encoder"
            
        speech_out = encoder_out[0][f"{key}_out"][0].transpose(0, 1) # S x B x D
        speech_lens = encoder_out[0][f"{key}_out_lengths"][0] # torch.Size([B])
        speech_padding_mask = encoder_out[0][f"{key}_padding_mask"][0] # B x S
        
        text_out = encoder_out[1]["encoder_out"][0]  # T x B x D
        text_lens = encoder_out[1]["src_lengths"][0].squeeze(-1) # torch.Size([B])
        text_padding_mask = encoder_out[1]["encoder_padding_mask"][0] # B x T

        S, B, _ = speech_out.size()
        T = text_out.size()[0]
        
        if speech_padding_mask is not None:
            non_padding_speech = ~speech_padding_mask # B x S
        else:
            non_padding_speech = (torch.ones(B, S) > 0).to(device=speech_out.device)
            
        if text_padding_mask is not None:
            non_padding_text = ~text_padding_mask # B x T
        else:
            non_padding_text = (torch.ones(B, T) > 0).to(device=text_out.device)
            
        if self.norm_before_ot:
            speech_out = speech_out / torch.linalg.norm(speech_out, dim=-1, keepdim=True)
            text_out = text_out / torch.linalg.norm(text_out, dim=-1, keepdim=True)
            
        if self.ot_positional_weight > 0.0: 
            speech_lens[speech_lens <= 1] = 2
            text_lens[text_lens <= 1] = 2
            # create tensor in which the elements are range of lengths
            speech_pos = torch.matmul(
                torch.tensor(range(S), dtype=torch.float, device=speech_out.device).unsqueeze(-1), 
                torch.ones((1, B), device=speech_out.device)
            ) # S x B
            text_pos = torch.matmul(
                torch.tensor(range(T), dtype=torch.float, device=speech_out.device).unsqueeze(-1), 
                torch.ones((1, B), device=speech_out.device)
            ) # T x B
            speech_pos = self.ot_positional_weight * speech_pos / (speech_lens - 1).unsqueeze(0) # S x B
            text_pos = self.ot_positional_weight * text_pos / (text_lens - 1).unsqueeze(0) # T x B
            speech_out = torch.cat((speech_out, speech_pos.unsqueeze(-1)), dim=-1)
            text_out = torch.cat((text_out, text_pos.unsqueeze(-1)), dim=-1)
            
        speech_weights = (
            torch.ones_like(non_padding_speech) / 
            torch.sum(non_padding_speech, dim=-1).unsqueeze(-1) *
            non_padding_speech
        )
        text_weights = (
            torch.ones_like(non_padding_text) / 
            torch.sum(non_padding_text, dim=-1).unsqueeze(-1) *
            non_padding_text
        )

        if self.detach_text_encoder:
            text_out = text_out.detach()
        
        with torch.cuda.amp.autocast(enabled=False):
            wass_loss = ot_loss(
                speech_weights.float(),
                speech_out.float().transpose(0, 1).contiguous(),
                text_weights.float(),
                text_out.float().transpose(0, 1).contiguous()
            )
            if torch.any(torch.isnan(wass_loss)):
                num_nans = torch.sum(torch.isnan(wass_loss))
                wass_loss = torch.nansum(wass_loss)
            else:
                num_nans = torch.tensor(0)
                wass_loss = torch.sum(wass_loss)
            
        return wass_loss, num_nans

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        wass_loss_sum = utils.item(
            sum(log.get("wass_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        if ctc_loss_sum != 0.0:
            metrics.log_scalar(
                "ctc_loss",
                ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if wass_loss_sum != 0:
            metrics.log_scalar(
                "wass_loss",
                wass_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
