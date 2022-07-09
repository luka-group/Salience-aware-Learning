# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/3/18 20:58
@Description: 
"""
import random
from dataclasses import dataclass
from typing import Optional, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


@dataclass
class JointDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    mask_threshold = 2.102626e-06
    top_k = 3

    # mask_threshold = 0.0025

    def __call__(self, features):

        # self._mask_label(features)
        self._replace_pos(features)
        features = self._replace_token(features)
        self._mask_label(features)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return features

    def _replace_token(self, features):
        if len(features[0]['replace_probs']) == 1:
            return features

        new_features = []
        for i in range(len(features)):
            # augmentation
            # scale = 1.0 / sum(features[i]['replace_probs'][:self.top_k])
            scale = 1.0
            features[i]['replace_probs'] = [x * scale for x in features[i]['replace_probs'][:self.top_k]]
            features[i]['replace_token_ids'] = features[i]['replace_token_ids'][:self.top_k]
            for tk, prob in zip(features[i]['replace_token_ids'], features[i]['replace_probs']):
                new_feature = {}
                for key, value in features[i].items():
                    new_feature[key] = value
                new_feature['replace_probs'] = prob
                new_feature['replace_token_ids'] = tk
                new_input = [x for x in features[i]['ver_input_ids']]
                new_input[features[i]['replace_pos']] = tk
                new_feature['ver_input_ids'] = new_input
                new_features.append(new_feature)

            # original data
            features[i]['replace_probs'] = 1.0
            features[i]['replace_token_ids'] = -100
            new_features.append(features[i])
        return new_features

    def _replace_pos(self, features):

        for i in range(len(features)):
            min_salient_score = 1.1
            masked_pos = -1
            for j, type_ids in enumerate(features[i]['sum_token_type_ids']):
                if type_ids[0] != 0:  # table token not masked
                    break
                elif features[i]['sum_labels'][j] in [0, 101, 102]:  # special token not masked
                    continue
                else:
                    sc = abs(features[i]['sum_token_salient_scores'][j])
                    if sc < min_salient_score:
                        masked_pos = j
                        min_salient_score = sc

            features[i]['replace_pos'] = masked_pos

    def _mask_label(self, features):

        for i in range(len(features)):
            if features[i]['ver_labels'] == 0 or features[i]['replace_token_ids'] != -100:
                features[i]['sum_labels'] = [-100 for _ in range(len(features[i]['sum_labels']))]
            else:
                max_salient_score = -1
                masked_pos = -1
                for j, type_ids in enumerate(features[i]['sum_token_type_ids']):
                    if type_ids[0] != 0:  # table token not masked
                        break
                    elif features[i]['sum_labels'][j] in [0, 101, 102]:  # special token not masked
                        continue
                    else:
                        if features[i]['sum_token_salient_scores'][j] > max_salient_score:
                            masked_pos = j
                            max_salient_score = features[i]['sum_token_salient_scores'][j]

                features[i]['sum_input_ids'][masked_pos] = 103

                for j, type_ids in enumerate(features[i]['sum_token_type_ids']):
                    if j != masked_pos:
                        features[i]['sum_labels'][j] = -100
