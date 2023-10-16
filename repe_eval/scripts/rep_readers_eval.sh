#!/bin/bash

source /opt/rh/devtoolset-10/enable

model_name_or_path=$1
task=$2
ntrain=$3
seed=$4
echo "model_name_or_path=$model_name_or_path"
echo "task=$task"
echo "ntrain=$ntrain"
echo "seed=$seed"


cd .. 
python rep_reading_eval.py \
    --model_name_or_path $model_name_or_path \
    --task $task \
    --ntrain $ntrain \
    --seed $seed 
