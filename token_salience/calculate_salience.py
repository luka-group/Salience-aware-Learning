# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/4/8 18:31
@Description: 
"""
import json
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm

data_file = '../data/train.jsonl'
ori_pred_file = 'results/origin_scores.txt'
mask_pred_file = 'results/masked_scores.txt'
out_file = 'results/salience_scores.json'


def softmax(a, b):
    return np.exp(a) / (np.exp(a) + np.exp(b))

ssid_to_sid = {}
ssid = 0
with jsonlines.open(data_file) as reader:
    for sid, obj in enumerate(tqdm(reader)):
        ssid_to_sid[ssid] = sid
        ssid += 1

ori_scores = {}
with open(ori_pred_file) as f:
    for sid, line in tqdm(enumerate(f.readlines())):
        elems = line.strip().split()
        refuted_score = float(elems[0])
        entailed_score = float(elems[1])
        confidence_score = softmax(refuted_score, entailed_score)
        assert confidence_score > 0
        assert confidence_score < 1
        ori_scores[sid] = confidence_score

token_scores = {}
salient_scores = {}
all_scores = []
ssid = 0
prev_batchid = None
with open(mask_pred_file) as f:
    for line in tqdm(f.readlines()):
        elems = line.strip().split()

        refuted_score = float(elems[0])
        entailed_score = float(elems[1])
        confidence_score = softmax(refuted_score, entailed_score)
        assert confidence_score > 0
        assert confidence_score < 1

        batchid = int(elems[2])
        if prev_batchid is None:
            prev_batchid = batchid
        if batchid != prev_batchid:
            ssid += 1
            prev_batchid = batchid
        sid = ssid_to_sid[ssid]

        tkid = int(elems[3])

        if sid not in token_scores:
            token_scores[sid] = {}
            salient_scores[sid] = {}
        token_scores[sid][tkid] = confidence_score
        salient_scores[sid][tkid] = ori_scores[sid] - confidence_score
        all_scores.append(salient_scores[sid][tkid])

with open(out_file, 'w') as f:
    json.dump(salient_scores, f)

pos = [0.05 * i for i in range(20)]
df = pd.Series(all_scores)
print(df.quantile(pos))
print(np.percentile(df, np.array(pos) * 100))
