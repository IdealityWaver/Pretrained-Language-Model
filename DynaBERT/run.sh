#!/bin/bash
#python eval_glue.py --model_type bert  --per_gpu_eval_batch_size 1 --task_name SST-2 --data_dir ~/odroid/SST-2/ --max_seq_length 128 --model_dir ./models/SST-2 --output_dir /tmp --depth_mult 1 --width_mult 0.5
TASK=SST-2
#TASK=STS-B
#TASK=MNLI
#for ((emb = 1; emb <= 5; emb++))

: '
for ((emb = 3; emb < 4; emb++))
do 
	for ((enc=2; enc <=6; enc++))
	do
		python eval_glue.py \
			--model_type bert \
			--per_gpu_eval_batch_size 128 \
			--task_name ${TASK} \
			--data_dir ../../${TASK}/ \
			--max_seq_length 128 \
			--model_dir /home/lwg/models/${TASK}/ \
			--output_dir /home/lwg/res \
			--depth_mult 1 \
			--width_mult 1 \
			--emb ${emb} \
			--enc ${enc} 
	done
done 
' 
#--model_dir /home/lwg/models \
#--model_dir ./models/${TASK} \
python eval_glue.py \
	--model_type bert \
	--per_gpu_eval_batch_size 128 \
       	--task_name ${TASK} \
	--data_dir ../../${TASK}/ \
       	--max_seq_length 128 \
	--model_dir /home/lwg/models/${TASK} \
       	--output_dir /tmp \
       	--depth_mult 1 \
	--width_mult 1 \
	--emb 0 \
	--enc 0

