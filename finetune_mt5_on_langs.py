import os
import subprocess

SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"
MODEL = "google/mt5-small"
LR = 0.001 # Same as mt5 paper

for lang in ["hu", "bo", "am", "sd", "ta", "tg", "vi"]: #os.listdir(f"{SCRIPTS_DIR}lang_splits/"):
    train_dataset = f"{SCRIPTS_DIR}lang_splits/{lang}"
    output_dir = f"{SCRIPTS_DIR}lang_models/{lang}"
    eval_command = f"sbatch \
                     {SCRIPTS_DIR}helper_scripts/run_finetune_t5.sh \
                     {MODEL} {train_dataset} {output_dir} {LR}"
    subprocess.run(eval_command, stdout=subprocess.PIPE, shell=True)
