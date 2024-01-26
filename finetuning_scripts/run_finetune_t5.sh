#!/bin/bash
#SBATCH --job-name=t5-time-vec-projections
#SBATCH --account=ark
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=["a40|a100|titan"]


MODEL=$1
TRAIN_DATASET=$2
OUT_DIR=$3
LR=$4 # 0.0008 for mt5-small
SEED=42
LM_PHRASE="--lm"
LORA_PHRASE=""

echo "Training ${MODEL} on ${TRAIN_DATASET}!"
python -u /mmfs1/gscratch/ark/knylund/yidlm-tests/finetuning_scripts/finetune_t5.py \
    --model_name_or_path $MODEL \
    --dataset_name $TRAIN_DATASET \
    --do_train \
    --do_eval \
    --input_column_1 "text" \
    --output_dir $OUT_DIR \
    --seed $SEED \
    --save_steps 200 \
    --save_strategy no \
    --source_prefix_1 "lm:" \
    --target_label label \
    --learning_rate $LR \
    --max_predict_samples 1000 \
    --max_source_length 128 \
    --max_target_length 128 \
    --gradient_accumulation_steps 8 \
    --ddp_find_unused_parameters False \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --patience 3 \
    $LM_PHRASE \
    $LORA_PHRASE