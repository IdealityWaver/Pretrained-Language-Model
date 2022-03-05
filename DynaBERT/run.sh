#!/bin/bash
#python eval_glue.py --model_type bert  --per_gpu_eval_batch_size 1 --task_name SST-2 --data_dir ~/odroid/SST-2/ --max_seq_length 128 --model_dir ./models/SST-2 --output_dir /tmp --depth_mult 1 --width_mult 0.5
python eval_glue.py --model_type bert  --task_name SST-2 --data_dir ~/odroid/SST-2/ --max_seq_length 128 --model_dir ./models/SST-2 --output_dir /tmp --depth_mult 1 --width_mult 1
