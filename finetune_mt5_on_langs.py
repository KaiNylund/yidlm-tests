import subprocess

SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"
TRAIN_LANGS = [
    "af", "sq", "am", "ar", "hy", "as", "az", "bn", "ba", "eu", "be", "br", "bg",
    "my", "ca", "ceb", "ckb", "ce", "zh", "cv", "hr", "cs", "da", "dv", "nl", "en",
    "eo", "et", "tl", "fi", "fr", "gl", "ka", "de", "el", "gu", "he", "hi", "hu",
    "is", "id", "ga", "it", "ja", "kn", "kk", "km", "ko", "ku", "ky", "lo", "la",
    "lv", "lt", "lb", "mk", "mg", "ms", "ml", "mr", "mn", "ne", "no", "nn", "or",
    "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sah", "sa", "sr", "sd", "si", "sk",
    "sl", "azb", "es", "sv", "tg", "ta", "tt", "te", "th", "bo", "tr", "uk", "ur",
    "ug", "uz", "vi", "cy", "fy", "pnb", "yi"
]
MODEL = "google/mt5-small"
LR = 0.0008

for lang in ["yi"]: #TRAIN_LANGS:
    train_dataset = f"{SCRIPTS_DIR}lang_splits/{lang}"
    output_dir = f"{SCRIPTS_DIR}lang_models/{lang}"
    eval_command = f"sbatch \
                     {SCRIPTS_DIR}finetuning_scripts/run_finetune_t5.sh \
                     {MODEL} {train_dataset} {output_dir} {LR}"
    subprocess.run(eval_command, stdout=subprocess.PIPE, shell=True)
