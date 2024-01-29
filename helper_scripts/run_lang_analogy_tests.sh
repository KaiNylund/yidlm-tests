#!/bin/bash
#SBATCH --job-name=mt5-small-lang-finetuning
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --partition=gpu-2080ti
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mmfs1/gscratch/ark/knylund/yidlm-tests/slurm_logs/%j.out
# --partition=ckpt
# --constraint=["a40|a100"]

python -u /mmfs1/gscratch/ark/knylund/yidlm-tests/lang_analogy_tests.py