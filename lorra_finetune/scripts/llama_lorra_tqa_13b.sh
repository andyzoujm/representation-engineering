#!/bin/bash

source /opt/rh/devtoolset-10/enable

ds_master_port=$((29000 + RANDOM % 1000))

cd ..

deepspeed --master_port $ds_master_port src/llama2_lorra.py \
    --model_name_or_path  "meta-llama/Llama-2-13b-chat-hf" \
    --user_tag '[INST]' \
    --assistant_tag '[/INST]' \
    --pos_type 'a truthful' \
    --neg_type 'an untruthful' \
    --control_template "Give {type} answer." \
    --target_layers '10,13,16,19,22,25,28,31,34,37' \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./lorra_tqa_13b \
    --overwrite_output_dir \
    --max_steps 70 \
    --fp16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 5  \
    --save_total_limit 0 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 128 \
    --q_lora False \
    --deepspeed configs/ds_zero1.json \
    --gradient_checkpointing True \
    --report_to none


# "/data/private_models/cais_models/llama-2/llama/llama-2-13b-chat-hf/"
# "/data/private_models/cais_models/vicuna/vicuna-30b-uncensored"