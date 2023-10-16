#!/bin/bash

model_sizes=("7b" "13b" "70b")
for model_size in "${model_sizes[@]}"; do
    for seed in {0..9}; do
        model_name_or_path="meta-llama/Llama-2-${model_size}-hf"
        gpus=1

        if [ "$model_size" = "70b" ]; then
            gpus=3
        fi

        # task="obqa"
        # ntrain=5

        # task="csqa"
        # ntrain=7

        task="arc_challenge"
        ntrain=25

        # task="arc_easy"
        # ntrain=25

        # task="race"
        # ntrain=3

        sbatch --nodes=1 --gpus-per-node=$gpus --time=48:00:00 --job-name="lat_bench" --output="$task/${model_size}_new/slurm-$task-$model_size-ntrain$ntrain-seed$seed-test-%j.out" rep_readers_eval.sh $model_name_or_path $task $ntrain $seed
    done
done