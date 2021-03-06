CUDA_VISIBLE_DEVICES=0 python run_joint_model.py \
--verifier_path google/tapas-large-finetuned-tabfact \
--model_mode joint \
--train_file ../data/train_with_salience_augmented.jsonl \
--validation_file ../data/val_with_salience_augmented.jsonl \
--do_train \
--do_eval \
--loss_ratio 0.5 \
--max_seq_length 512 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 16 \
--learning_rate 5e-5 \
--max_steps 10000 \
--evaluation_strategy steps \
--warmup_steps 10 \
--eval_steps  10 \
--logging_steps  10 \
--save_steps 10 \
--load_best_model_at_end \
--metric_for_best_model accuracy \
--dataloader_num_workers 8 \
--overwrite_cache \
--overwrite_output_dir \
--output_dir ./outputs/
