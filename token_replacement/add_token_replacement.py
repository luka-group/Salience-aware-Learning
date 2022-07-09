# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/4/8 20:39
@Description: 
"""
import json
import jsonlines
from tqdm import tqdm
import numpy as np

data_file = '../data/train_with_salience.jsonl'
new_data_file = '../data/train_with_salience_augmented.jsonl'

pred_logits = np.load('results/preds_logits.npy')
pred_token_ids = np.load('results/preds_token_ids.npy')

with jsonlines.open(data_file) as reader, open(new_data_file, 'w') as f:
    for i, obj in tqdm(enumerate(reader)):

        obj['replace_probs'] = pred_logits[i].tolist()
        obj['replace_token_ids'] = pred_token_ids[i].tolist()

        json.dump(obj, f)
        f.write('\n')

# with jsonlines.open('data/val_salient.json') as reader, open('data/val_salient_replace.json', 'w') as f:
#     for i, obj in tqdm(enumerate(reader)):
#
#         obj['replace_probs'] = [1.0]
#         obj['replace_token_ids'] = [-100]
#
#         json.dump(obj, f)
#         f.write('\n')


# with jsonlines.open('data/test_salient_wid.json') as reader, open('data/test_salient_replace_wid.json', 'w') as f:
#     for i, obj in tqdm(enumerate(reader)):
#
#         obj['replace_probs'] = [1.0]
#         obj['replace_token_ids'] = [-100]
#
#         json.dump(obj, f)
#         f.write('\n')
