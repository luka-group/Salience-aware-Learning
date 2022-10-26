# Salience-aware-Learning

Code for our paper [Table-based Fact Verification with Salience-aware Learning](https://arxiv.org/abs/2109.04053) at EMNLP 2021 Findings.

## Installation
```bash
pip install -r requirements.txt
```
Install [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter).

## Data
We conduct experiments on the [TabFact](https://tabfact.github.io/) dataset.
The statements in officially released train/val/test set are lemmatized. 
We use the raw (unlemmatized) statements.
More discussion can be found in this [issue](https://github.com/NielsRogge/Transformers-Tutorials/issues/2).

Download the [train/val/test set](https://onedrive.live.com/?authkey=%21AKeWiSjW2BYUsmY&id=2CFE0E4E795F88D9%2141297&cid=2CFE0E4E795F88D9) to `./data`.

Download the [table set](https://github.com/wenhuchen/Table-Fact-Checking/tree/master/data/all_csv) to `./data/tables`.

To convert raw data to model inputs:
```bash
cd data
python preprocess.py
```

## Token Salience Detection
```bash
cd token_salience
```
* First, run `bash run_origin.sh` to get predictions for original inputs.
* Second, run `bash run_masked.sh` to get predictions for inputs with masked tokens.
* Third, run `python calculate_salience.py` to get salience scores by comparing the outputs of last two steps.
* Finally, run `python add_salience_to_data.py` to merge the salience scores into input data.

## Non-salient Token Replacement
```bash
cd token_replacement
```
* First, run `bash run_mlm.sh` to get predictions for replacing non-salient tokens.
* Second, run `python add_token_replacement.py` to merge the token replacement candidates into input data.

## Joint Fact Verification and Salient Token Prediction
```bash
cd joint_model
bash run_joint_model.sh
```
