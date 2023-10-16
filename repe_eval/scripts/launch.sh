#!/bin/bash

# We manually choosing seed similiar to the average runs reported in the paper here
# To correctly reproduce the results reported in the paper one should do 
# 'for seed in {0..9}' for each task and model (see launch_seeds.sh)


##################################### OBQA #####################################
task="obqa"
ntrain=5
seed=3

model_name_or_path="meta-llama/Llama-2-7b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-13b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-70b-hf"
sbatch --nodes=1 --gpus-per-node=2 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed


# ##################################### CSQA #####################################
task="csqa"
ntrain=7
seed=5

model_name_or_path="meta-llama/Llama-2-7b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-13b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-70b-hf"
sbatch --nodes=1 --gpus-per-node=2 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed


# ##################################### ARC-easy #####################################
task="arc_easy"
ntrain=25
seed=1

model_name_or_path="meta-llama/Llama-2-7b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-13b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-70b-hf"
sbatch --nodes=1 --gpus-per-node=2 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed


# ##################################### ARC-challenge #####################################
task="arc_challenge"
ntrain=25
seed=1

model_name_or_path="meta-llama/Llama-2-7b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-13b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-70b-hf"
sbatch --nodes=1 --gpus-per-node=2 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed


# ##################################### RACE #####################################
task="race"
ntrain=3
seed=0

model_name_or_path="meta-llama/Llama-2-7b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-13b-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-70b-hf"
sbatch --nodes=1 --gpus-per-node=3 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed


# ##################################### TQA #####################################
task="tqa"
ntrain=0
seed=2

model_name_or_path="meta-llama/Llama-2-7b-chat-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-13b-chat-hf"
sbatch --nodes=1 --gpus-per-node=1 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

model_name_or_path="meta-llama/Llama-2-70b-chat-hf"
sbatch --nodes=1 --gpus-per-node=2 --output="slurm-$task-ntrain$ntrain-seed$seed-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed

