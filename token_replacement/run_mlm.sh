CUDA_VISIBLE_DEVICES=0 python mlm.py \
--model_name_or_path bert-large-uncased \
--test_file ../data/train_with_salience.jsonl \
--output_dir ./results \
--do_predict
