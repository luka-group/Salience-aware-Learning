# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/4/20 22:49
@Description: 
"""
import torch
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.modeling_outputs import dataclass, ModelOutput, Optional, Tuple

@dataclass
class MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    token_ids: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SimpleMaskedLM(BertForMaskedLM):

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
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        masked_preds = []
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                if input_ids[i][j] == 103:
                    masked_preds.append(prediction_scores[i][j])
                    break
        masked_preds = torch.stack(masked_preds)

        sf = torch.nn.Softmax(dim=1)
        masked_preds = sf(masked_preds)

        sorted, indices = torch.sort(masked_preds, dim=1, descending=True)

        return MaskedLMOutput(
            loss=torch.FloatTensor([0]),
            logits=sorted[:, :10],
            token_ids=indices[:, :10],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
