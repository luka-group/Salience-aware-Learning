# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/3/13 19:05
@Description: 
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.generation_utils import Dict, Any, ModelOutput
from transformers import PreTrainedModel, AutoConfig
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.models.tapas.modeling_tapas import TapasForSequenceClassification, TapasPreTrainedModel, TapasConfig, \
    TapasModel, MaskedLMOutput, PreTrainedModel, TapasForMaskedLM
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel, EncoderDecoderConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput
from transformers.file_utils import ModelOutput

TOPK = 3


@dataclass
class JointOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    ver_loss = None
    sum_loss = None
    logits: torch.FloatTensor = None
    # ver_logits: torch.FloatTensor = None
    # sum_logits: torch.FloatTensor = None


class NewTapasForSequenceClassification(TapasForSequenceClassification):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            replace_probs=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                net_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                weighted_loss = net_loss.view(1, -1).mm(replace_probs.view(-1, 1))
                loss = weighted_loss[0][0] / torch.sum(replace_probs)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NewTapasForMaskedLM(TapasForMaskedLM):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            replace_probs=None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class JointConfig(TapasConfig):
    def __init__(
            self,
            loss_ratio=0.5,
            mode='ver',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.loss_ratio = loss_ratio
        self.mode = mode


class JointModel(PreTrainedModel):
    config_class = JointConfig

    def __init__(self, joint_config=None, verifier_path=None, classifier_config=None):
        super().__init__(joint_config)
        self.config = joint_config

        verifier_path = 'google/tapas-large-finetuned-tabfact'

        self.fact_verifier = NewTapasForSequenceClassification.from_pretrained(verifier_path)

        self.cell_predictor = NewTapasForMaskedLM.from_pretrained(verifier_path)

        self.cell_predictor.tapas = self.fact_verifier.tapas
        self.cell_predictor.tie_weights()

        self.mode = joint_config.mode
        # if joint_config.freeze_encoder:
        #     for name, param in self.cell_predictor.tapas.named_parameters():
        #         param.requires_grad = False

        self.alpha = joint_config.loss_ratio

    def forward(
            self,
            ver_input_ids=None,
            ver_attention_mask=None,
            ver_token_type_ids=None,
            ver_labels=None,
            ver_position_ids=None,
            ver_head_mask=None,
            ver_inputs_embeds=None,
            ver_output_attentions=None,
            ver_output_hidden_states=None,
            ver_return_dict=None,

            replace_probs=None,
            replace_token_ids=None,
            sum_token_salient_scores=None,

            sum_input_ids=None,
            sum_attention_mask=None,
            sum_token_type_ids=None,
            sum_labels=None,
            sum_decoder_input_ids=None,
            sum_decoder_attention_mask=None,
            sum_encoder_outputs=None,
            sum_past_key_values=None,
            sum_inputs_embeds=None,
            sum_decoder_inputs_embeds=None,
            sum_use_cache=None,
            sum_output_attentions=None,
            sum_output_hidden_states=None,
            sum_return_dict=None,
            **kwargs,
    ):
        ver_output = None
        if self.mode in ['joint', 'ver']:
            ver_output = self.fact_verifier(
                input_ids=ver_input_ids,
                attention_mask=ver_attention_mask,
                token_type_ids=ver_token_type_ids,
                position_ids=ver_position_ids,
                head_mask=ver_head_mask,
                inputs_embeds=ver_inputs_embeds,
                labels=ver_labels,
                output_attentions=ver_output_attentions,
                output_hidden_states=ver_output_hidden_states,
                return_dict=ver_return_dict,
                replace_probs=replace_probs
            )

        sum_output = None
        if self.mode in ['joint', 'sum'] and sum_input_ids is not None:
            sum_output = self.cell_predictor(
                input_ids=sum_input_ids,
                attention_mask=sum_attention_mask,
                token_type_ids=sum_token_type_ids,
                decoder_input_ids=sum_decoder_input_ids,
                decoder_attention_mask=sum_decoder_attention_mask,
                encoder_outputs=sum_encoder_outputs,
                past_key_values=sum_past_key_values,
                inputs_embeds=sum_inputs_embeds,
                decoder_inputs_embeds=sum_decoder_inputs_embeds,
                labels=sum_labels,
                use_cache=sum_use_cache,
                output_attentions=sum_output_attentions,
                output_hidden_states=sum_output_hidden_states,
                return_dict=sum_return_dict,
                replace_probs=replace_probs
            )

        if ver_output is not None:
            loss = ver_output["loss"]
            logits = ver_output["logits"]
            if sum_output is not None:
                loss = (1 - self.alpha) * loss + self.alpha * sum_output["loss"]
        else:
            loss = sum_output["loss"]
            logits = sum_output["logits"]

        return JointOutput(
            loss=loss,
            # ver_loss=ver_output["loss"],
            # sum_loss=sum_output["loss"] if sum_output else None,
            logits=logits
        )
