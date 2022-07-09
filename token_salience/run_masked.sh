CUDA_VISIBLE_DEVICES=0 python masked.py \
--model_name_or_path google/tapas-large-finetuned-tabfact \
--test_file ../data/train_processed.jsonl \
--do_predict \
--max_seq_length 512 \
--per_device_eval_batch_size 128 \
--overwrite_cache \
--output_dir ./results
