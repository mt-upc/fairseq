# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
import editdistance

from fairseq import metrics, utils
from fairseq.data.data_utils import lengths_to_padding_mask
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
        default=0.0,
        metadata={"help": "Weight for CTC loss"},
    )
    ot_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for OT loss"},
    )
    ot_pos_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for OT positional embeddings"},
    )
    ot_distribution: ChoiceEnum(["uniform", "norm"]) = field(
        default="uniform",
        metadata={"help": "distribution of the weights in the OT loss"},
    )
    ot_loss: SAMPLE_LOSS_CHOICES = field(
        default="sinkhorn",
        metadata={"help": "type of distance measure between X_i and Y_j"},
    ) # TODO change this to energy
    ot_p: int = field(
        default=2,
        metadata={"help": "p in SampleLoss"},
    ) # TODO change this to 1
    ot_blur: float = field(
        default=0.05,
        metadata={"help": "blur in SampleLoss"},
    )
    ot_scaling: float = field(
        default=0.5,
        metadata={"help": "scaling in SampleLoss"},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "debug mode, save stuff during eval."},
    )
    debug_save_dir: str = field(
        default="",
        metadata={"help": "debug save dir."},
    )
    ot_aux_layers: str = field(
        default="",
        metadata={"help": "comma-separated auxiliary layers for OT loss."},
    )
    ot_aux_weights: str = field(
        default="",
        metadata={"help": "comma-separated auxiliary weights for OT loss."},
    )
    eval_wer: bool = field(
        default=False,
        metadata={"help": "report WER during eval."},
    )
    
@register_criterion(
    "ctc_wass", dataclass=CtcWassersteinCriterionConfig
)
class CtcWassersteinCriterion(CtcCriterion):
    def __init__(self, cfg: CtcWassersteinCriterionConfig, task: FairseqTask):
        super().__init__(cfg, task)
        
        self.sep_idx = task.target_dictionary.index("|")
        
        assert self.blank_idx == self.pad_idx

        self.ctc_weight = cfg.ctc_weight
        self.ot_weight = cfg.ot_weight
        self.ot_distribution = cfg.ot_distribution
        self.ot_loss = cfg.ot_loss
        self.ot_p = cfg.ot_p
        self.ot_blur = cfg.ot_blur
        self.ot_scaling = cfg.ot_scaling
        self.ot_pos_weight = cfg.ot_pos_weight
        
        self.ot_aux_layers = [int(l) for l in cfg.ot_aux_layers.split(",")] if cfg.ot_aux_layers else []
        self.ot_aux_weights = [float(w) for w in cfg.ot_aux_weights.split(",")] if cfg.ot_aux_weights else []
        assert len(self.ot_aux_layers) == len(self.ot_aux_weights)
        
        self.eval_wer = cfg.eval_wer
        
        if hasattr(cfg, "debug"):
            self.save = cfg.debug
            self.debug_save_dir = Path(cfg.debug_save_dir)
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save = False

        logging.info(f"*** Loss function ***")
        logging.info(f"ctc_weight = {self.ctc_weight}")
        logging.info(f"ot_weight = {self.ot_weight}")
        logging.info(f"ot_pos_weight = {self.ot_pos_weight}")
        logging.info(f"ot_loss = {self.ot_loss}, ot_p = {self.ot_p}, ot_blur = {self.ot_blur}, ot_scaling = {self.ot_scaling}")

        self.ot_loss = SamplesLoss(
            loss=cfg.ot_loss, p=self.ot_p, blur=self.ot_blur, scaling=self.ot_scaling
        )

    def forward(self, model, sample):
        net_input = sample["net_input"]
        net_output, encoder_out = model(
            src_tokens=net_input["src_tokens"],
            src_lengths=net_input["src_lengths"],
            src_txt_tokens=net_input["src_txt_tokens"] if "src_txt_tokens" in net_input else None,
            src_txt_lengths=net_input["src_txt_lengths"] if "src_txt_lengths" in net_input else None,
        )
        sample_size = (
            net_input["src_tokens"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        loss = 0.0
        extra = {"ctc_loss": 0.0, "wass_loss": 0.0}
        for layer_id in self.ot_aux_layers:
            extra[f"wass_loss_{layer_id}"] = 0.0

        if self.ctc_weight > 0.0:
            ctc_loss, extra = self.compute_ctc_loss(
                model, net_output, encoder_out, sample["target"], extra
            )
            loss += self.ctc_weight * ctc_loss

<<<<<<< HEAD
        lvl = None
        if self.ot_weight > 0.0:
            lvl = -1
            wass_loss = self.compute_wass_loss(encoder_out, net_input, lvl=lvl, ids=sample["example_id"])
            loss += self.ot_weight * wass_loss
            extra["wass_loss"] = wass_loss
        elif self.ot_emb_weight > 0.0 and not model.training:
            lvl = -1
            speech_out = model.encoder.forward_context(encoder_out[0])
            encoder_out = (speech_out, encoder_out[1])
            wass_loss = self.compute_wass_loss(encoder_out, net_input, lvl=lvl, ids=sample["example_id"])
            extra["wass_loss"] = wass_loss

        if self.ot_emb_weight > 0.0:
            lvl = 0
            wass_emb_loss = self.compute_wass_loss(encoder_out, net_input, lvl=lvl, ids=sample["example_id"])
            loss += self.ot_emb_weight * wass_emb_loss
            extra["wass_emb_loss"] = wass_emb_loss
            
        if self.ot_mid_weight > 0.0:
            lvl = 6
            wass_mid_loss = self.compute_wass_loss(encoder_out, net_input, lvl=lvl, ids=sample["example_id"])
            loss += self.ot_mid_weight * wass_mid_loss
            extra["wass_mid_loss"] = wass_mid_loss
            
=======
        speech_out, speech_lens, speech_padding_mask = self._get_speech_repr(encoder_out, lvl=-1)
        text_out, text_lens, text_padding_mask = self._get_text_repr(net_input, encoder_out, lvl=-1)

        m = len(self.ot_aux_weights) + 1
        B = speech_out.size(1)
        
        speech_out = torch.cat([speech_out, *[encoder_out[0]["context_ln_results"][layer_id] for layer_id in self.ot_aux_layers]], dim=1)
        speech_lens = speech_lens.repeat(m)
        speech_padding_mask = speech_padding_mask.repeat(m, 1)
        
        if "src_txt_enc" in net_input:
            text_out = torch.cat([text_out, *[net_input["src_txt_ln_results"][layer_id].transpose(0, 1) for layer_id in self.ot_aux_layers]], dim=1)
        else:   
            text_out = torch.cat([text_out, *[encoder_out[1]["ln_results"][layer_id] for layer_id in self.ot_aux_layers]], dim=1)
        text_lens = text_lens.repeat(m)
        text_padding_mask = text_padding_mask.repeat(m, 1)
        
        wass_loss = self.compute_wass_loss(speech_out, speech_lens, speech_padding_mask, text_out, text_lens, text_padding_mask, ids=sample["example_id"])
        
        extra["wass_loss"] = wass_loss[:B].sum()
        loss += self.ot_weight * extra["wass_loss"]
        
        for i, layer_id in enumerate(self.ot_aux_layers):
            extra[f"wass_loss_{layer_id}"] = wass_loss[B*(i+1):B*(i+2)].sum()
            loss += self.ot_aux_weights[i] * extra[f"wass_loss_{layer_id}"]
>>>>>>> zero-shot-st_batched
    
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
        }
        
        for i, layer_id in enumerate(self.ot_aux_layers):
            logging_output[f"wass_loss_{layer_id}"] = utils.item(extra[f"wass_loss_{layer_id}"].data) if extra[f"wass_loss_{layer_id}"] != 0.0 else 0.0
        
        if net_output is not None:
            if "compression_rate" in encoder_out[0]:
                logging_output["compression_rate"] = utils.item(
                    encoder_out[0]["compression_rate"].data.sum()
                )
            if "char_compression_rate" in encoder_out[0]:
                logging_output["char_compression_rate"] = utils.item(
                    encoder_out[0]["char_compression_rate"].data.sum()
                )
            if "token_compression_rate" in encoder_out[0]:
                logging_output["token_compression_rate"] = utils.item(
                    encoder_out[0]["token_compression_rate"].data.sum()
                )
                
        logging_output["speech_text_len_ratio"] = utils.item(
            (speech_lens[:B].float() / text_lens[:B].float()).data.sum()
        )

        if "num_predicted_sep" in extra:
            logging_output["num_predicted_sep"] = extra["num_predicted_sep"]
            logging_output["num_correct_sep"] = extra["num_correct_sep"]

        if not model.training and extra["ctc_loss"] != 0.0 and self.eval_wer:
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
        input_lengths = spch_encoder_out["encoder_out_lengths"][0]

        pad_mask = (targets != self.blank_idx) & (targets != self.eos_idx)
        targets_flat = targets.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        # separator stats
        sep_preds = (lprobs.argmax(-1) == self.sep_idx).long()
        num_predicted_sep = torch.tensor(0, device=sep_preds.device, dtype=torch.long)
        for i in range(sep_preds.size(1)):
            # real number of prerdicted separators
            num_predicted_sep += sep_preds[:, i].unique_consecutive().sum()
        num_correct_sep = (targets_flat == self.sep_idx).sum()

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
        extra["num_predicted_sep"] = num_predicted_sep
        extra["num_correct_sep"] = num_correct_sep

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
    
    def _get_speech_repr(self, encoder_out, lvl=-1):
        
        if lvl == -1:
            speech_out = encoder_out[0]["context_out"][0].transpose(0, 1)  # S x B x D
        else:
            assert lvl in range(len(encoder_out[0]["context_ln_results"]))
            speech_out = encoder_out[0]["context_ln_results"][lvl]  # S x B x D
            
        speech_lens = encoder_out[0]["context_out_lengths"][0] # torch.Size([B])
        speech_padding_mask = encoder_out[0]["context_padding_mask"][0] # B x S
        
        if speech_padding_mask is None:
            speech_padding_mask = lengths_to_padding_mask(speech_lens)
        
        return speech_out, speech_lens, speech_padding_mask
    
    def _get_text_repr(self, net_input, encoder_out, lvl=-1):
        
        if "src_txt_enc" in net_input:
            if lvl == -1:
                text_out = net_input["src_txt_enc"].transpose(0, 1) # T x B x D
            else:
                assert lvl in range(len(net_input["src_txt_ln_results"]))
                text_out = net_input["src_txt_ln_results"][lvl].transpose(0, 1)  # T x B x D
                
            text_lens = net_input["src_txt_lengths"] # torch.Size([B])
            text_padding_mask = lengths_to_padding_mask(text_lens) # B x T
        else:
            if lvl == -1:
                text_out = encoder_out[1]["encoder_out"][0]  # T x B x D
            else:
                assert lvl in range(len(encoder_out[1]["ln_results"]))
                text_out = encoder_out[1]["ln_results"][lvl]  # T x B x D

            text_lens = encoder_out[1]["src_lengths"][0].squeeze(-1) # torch.Size([B])
            text_padding_mask = encoder_out[1]["encoder_padding_mask"][0] # B x T
            
            if text_padding_mask is None:
                text_padding_mask = lengths_to_padding_mask(text_lens)
        
        return text_out, text_lens, text_padding_mask
    
    def compute_wass_loss(self, speech_out, speech_lens, speech_padding_mask, text_out, text_lens, text_padding_mask, ids=None):

        S, B, _ = speech_out.size()
        T = text_out.size()[0]
        dev = speech_out.device
            
        # zero-out padding (remove later)
        speech_out = speech_out.masked_fill(speech_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        text_out = text_out.masked_fill(text_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)

        if self.ot_pos_weight > 0.0:
            # create tensor in which the elements are range of lengths
            speech_pos = torch.matmul(
                torch.tensor(range(S), dtype=torch.float, device=dev).unsqueeze(-1), 
                torch.ones((1, B), device=dev)
            ) # S x B
            text_pos = torch.matmul(
                torch.tensor(range(T), dtype=torch.float, device=dev).unsqueeze(-1), 
                torch.ones((1, B), device=dev)
            ) # T x B
            speech_pos = self.ot_pos_weight * speech_pos / (speech_lens - 1).unsqueeze(0) # S x B
            text_pos = self.ot_pos_weight * text_pos / (text_lens - 1).unsqueeze(0) # T x B
            speech_out = torch.cat((speech_out, speech_pos.unsqueeze(-1)), dim=-1) # S x B x D+1
            text_out = torch.cat((text_out, text_pos.unsqueeze(-1)), dim=-1) # T x B x D+1

        if self.ot_distribution == "uniform":
            speech_weights = torch.ones_like(speech_padding_mask) / speech_lens.unsqueeze(-1) # B x S
            text_weights = torch.ones_like(text_padding_mask) / text_lens.unsqueeze(-1) # B x T
        elif self.ot_distribution == "norm":
            speech_norm = torch.norm(speech_out.detach().transpose(0, 1), dim=-1) # B x S
            speech_weights = speech_norm / speech_norm.sum(dim=-1, keepdim=True) # B x S
            text_norm = torch.norm(text_out.detach().transpose(0, 1), dim=-1) # B x T
            text_weights = text_norm / text_norm.sum(dim=-1, keepdim=True) # B x T
        # zero weights for padding
        speech_weights.masked_fill_(speech_padding_mask, 0.0)
        text_weights.masked_fill_(text_padding_mask, 0.0)
            
        with torch.cuda.amp.autocast(enabled=False):
            wass_loss = self.ot_loss(
                speech_weights.float(),
                speech_out.float().transpose(0, 1).contiguous(),
                text_weights.float(),
                text_out.float().transpose(0, 1).contiguous()
            )
            
        if self.save:
            for i in range(B):
                torch.save(
                    {
                        "speech_out": speech_out[:speech_lens[i], i, :-1].detach().cpu().clone(),
                        "text_out": text_out[:text_lens[i], i, :-1].detach().cpu().clone(),
                        "wass_loss": wass_loss[i].detach().cpu().clone()
                    }, 
                    self.debug_save_dir / f"{ids[i]}_last.pt"
                )
        
        return wass_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
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
            
        for k in logging_outputs[0].keys():
            if k.startswith("wass_loss"):
                wass_loss_sum = utils.item(sum(log.get(k, 0) for log in logging_outputs))
                if wass_loss_sum != 0:
                    metrics.log_scalar(
                        k,
                        wass_loss_sum / sample_size / math.log(2),
                        sample_size,
                        round=3,
                    )

        num_predicted_sep = utils.item(
            sum(log.get("num_predicted_sep", 0) for log in logging_outputs)
        )
        num_correct_sep = utils.item(
            sum(log.get("num_correct_sep", 0) for log in logging_outputs)
        )
        sep_pred_ratio = num_predicted_sep / num_correct_sep
        metrics.log_scalar("sep_pred_ratio", sep_pred_ratio, round=3)        
        
        speech_text_len_ratio = utils.item(
            sum(log.get("speech_text_len_ratio", 0) for log in logging_outputs)
        ) / nsentences
        metrics.log_scalar("speech_text_len_ratio", speech_text_len_ratio, round=3)

        compression_rate = utils.item(
            sum(log.get("compression_rate", 0) for log in logging_outputs)
        )
        metrics.log_scalar("compression_rate", compression_rate / nsentences, round=3)
        char_compression_rate = utils.item(
            sum(log.get("char_compression_rate", 0) for log in logging_outputs)
        )
        metrics.log_scalar("char_compression_rate", char_compression_rate / nsentences, round=3)
        token_compression_rate = utils.item(
            sum(log.get("token_compression_rate", 0) for log in logging_outputs)
        )
        metrics.log_scalar("token_compression_rate", token_compression_rate / nsentences, round=3)

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        
        if "wer" in logging_outputs[0]:
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
