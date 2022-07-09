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

salience_file = 'results/salience_scores.json'
data_file = '../data/train_processed.jsonl'
new_data_file = '../data/train_with_salience.jsonl'

with open(salience_file) as f:
    token_salient_scores = json.load(f)

with jsonlines.open(data_file) as reader, open(new_data_file, 'w') as f:
    for sid, obj in tqdm(enumerate(reader)):
        token_scores = [-100.0 for _ in range(len(obj['ver_input_ids']))]

        sid = str(sid)

        if sid not in token_salient_scores:
            print(sid)

        if sid in token_salient_scores:
            for tkid, tkscore in token_salient_scores[sid].items():
                token_scores[int(tkid)] = tkscore

        obj['token_salient_scores'] = token_scores

        json.dump(obj, f)
        f.write('\n')
