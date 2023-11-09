#!/bin/bash

# sbatch --nodes=1 --gpus-per-node=1 --time=02:00:00 --output="slurm-lorra_tqa_13b-%j.out" llama_lorra_tqa_13b.sh

sbatch --nodes=1 --gpus-per-node=1 --partition=cais --time=02:00:00 --output="slurm-lorra_tqa_7b-%j.out" llama_lorra_tqa_7b.sh

