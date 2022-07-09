# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2021/3/14 19:11
@Description: 
"""
import os
import json
import jsonlines
from tqdm import tqdm
from pandas import DataFrame
from transformers import AutoTokenizer


def load_tableset(table_dir="tables"):
    tables = {}
    for file_name in tqdm(os.listdir(table_dir), desc="loading tables"):
        table_id = file_name
        file_path = os.path.join(table_dir, file_name)
        table = []
        with open(file_path) as f:
            for line in f.readlines():
                row = line.strip().split('#')
                table.append(row)
            tables[table_id] = DataFrame(table[1:]).astype(str)
            tables[table_id].columns = table[0]
    return tables


def load_and_export(file, fn):
    tapas_tokenizer = AutoTokenizer.from_pretrained("google/tapas-large", use_fast=True)

    with jsonlines.open(file) as reader, open(fn + "_processed.jsonl", "w") as f:
        for obj in tqdm(reader):
            sample = {
                'ver_input_ids': None, 'ver_attention_mask': None, 'ver_token_type_ids': None, "ver_label": None,
                'table_id': None
            }

            tid = obj['table_id']
            statement = obj['statement']
            label = obj['label']
            tokneized_seq = tapas_tokenizer(tables[tid], statement, padding=True, max_length=512, truncation=True)
            for key in tokneized_seq:
                sample['ver_' + key] = tokneized_seq[key]
            sample["ver_label"] = label
            sample["table_id"] = tid

            json.dump(sample, f)
            f.write('\n')

    return


if __name__ == "__main__":
    train_file = "train.jsonl"
    val_file = "val.jsonl"
    test_file = "test.jsonl"

    data_files = {
        "train": train_file,
        "val": val_file,
        "test": test_file,
    }

    tables = load_tableset()

    for key, value in data_files.items():
        load_and_export(value, key)
