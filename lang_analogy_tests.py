import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM
from lang_model_distances import cos_dist, get_model_dist
from helper_scripts.get_task_vector import get_task_vector, task_op


SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"
PRETRAINED_MODEL = "google/mt5-small"
PARAMS = ["all", "embeddings", "ff_layers", "attn"]

if __name__ == "__main__":
    out_dir = f"{SCRIPTS_DIR}mt5-small_lang_analogy_cos_sims"
    model_langs = []
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    for lang in os.listdir(f"{SCRIPTS_DIR}lang_models/"):
        if os.listdir(f"{SCRIPTS_DIR}lang_models/{lang}"):
            model_langs.append(lang)

    en_vec = get_task_vector(pretrained_model, AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/en").eval(), alpha=1.0)
    he_vec = get_task_vector(pretrained_model, AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/he").eval(), alpha=1.0)
    yi_vec = get_task_vector(pretrained_model, AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/yi").eval(), alpha=1.0)
    en_to_he_vec = task_op(he_vec, en_vec, op="subtract")
    en_to_yi_vec = task_op(yi_vec, en_vec, op="subtract")
    he_to_yi_vec = task_op(yi_vec, he_vec, op="subtract")

    analogy_cos_sims = defaultdict(dict)
    for lang in tqdm(model_langs):
        lang_vec = get_task_vector(pretrained_model, AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/{lang}").eval(), alpha=1.0)
        en_to_lang_vec = task_op(lang_vec, en_vec, op="subtract")
        lang_to_yi_vec = task_op(yi_vec, lang_vec, op="subtract")
        lang_to_he_vec = task_op(he_vec, lang_vec, op="subtract")
        if lang != "yi":
            analogy_cos_sims["en_to_he"][f"{lang}_to_yi"] = get_model_dist(en_to_he_vec, lang_to_yi_vec, dist_func=cos_dist, params=PARAMS)
        if lang != "en":
            analogy_cos_sims["he_to_yi"][f"en_to_{lang}"] = get_model_dist(he_to_yi_vec, en_to_lang_vec, dist_func=cos_dist, params=PARAMS)
        if lang != "he":
            analogy_cos_sims["en_to_yi"][f"{lang}_to_he"] = get_model_dist(en_to_yi_vec, lang_to_he_vec, dist_func=cos_dist, params=PARAMS)
    np.save(out_dir, analogy_cos_sims)
