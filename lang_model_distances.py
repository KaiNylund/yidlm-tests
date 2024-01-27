from transformers import AutoModelForSeq2SeqLM
from collections import defaultdict
from tqdm import tqdm
from helper_scripts.get_task_vector import get_task_vector
import numpy as np
import torch
import os


def cos_dist(A, B, axis=None):
    return np.dot(A, B) / (np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis))

# Assumes both models have the same params
def get_model_dist(m1, m2, dist_func, param="all", axis=None):
    m1_sd = m1.state_dict()
    m2_sd = m2.state_dict()
    m1_weights = []
    m2_weights = []
    with torch.no_grad():
        if param == "all":
            param_names = m1_sd.keys()
        elif param == "embeddings":
            param_names = ["shared.weight"]
        elif param == "ff_layers":
            param_names = [p for p in list(m1_sd.keys()) if "DenseReluDense" in p]
        elif param == "attn":
            param_names = [p for p in list(m1_sd.keys()) if "Attention" in p]
        elif param == "norm":
            param_names = [p for p in list(m1_sd.keys()) if "norm" in p]
        else:
            param_names = [param]
        for param_name in param_names:
            m1_weights.append(m1_sd[param_name].detach().numpy().flatten())
            m2_weights.append(m2_sd[param_name].detach().numpy().flatten())
        vec_dist = dist_func(np.hstack(np.array(m1_weights)),
                             np.hstack(np.array(m2_weights)), axis=axis)
        del m1_sd, m2_sd, m1_weights, m2_weights
        return vec_dist
    

SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"
PRETRAINED_MODEL = "google/mt5-small"
PARAMS = ["all", "embeddings", "ff_layers", "attn", "norm"]

if __name__ == "__main__":
    lang_model_paths = []
    model_langs = []
    for lang in os.listdir(f"{SCRIPTS_DIR}lang_splits/"):
            if os.listdir(f"{SCRIPTS_DIR}lang_splits/{lang}"):
                lang_model_paths.append(f"{SCRIPTS_DIR}lang_splits/{lang}")
                model_langs.append(lang)

    out_dir = f"{SCRIPTS_DIR}mt5-small_lang_model_cos_sims"
    sims_dict = defaultdict(lambda: defaultdict(dict))
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    for i, lang1 in tqdm(enumerate(model_langs)):
        lang1_model = AutoModelForSeq2SeqLM.from_pretrained(lang_model_paths[i]).eval()
        lang1_vec = get_task_vector(pretrained_model, lang1_model, alpha=1.0)
        del lang1_model
        for j, lang2 in enumerate(model_langs):
            if lang1 == lang2:
                for param in PARAMS:
                    sims_dict[lang1][lang2][param] = 1.0
            else:
                lang2_model = AutoModelForSeq2SeqLM.from_pretrained(lang_model_paths[j]).eval()
                lang2_vec = get_task_vector(pretrained_model, lang2_model, alpha=1.0)
                del lang2_model
                for param in PARAMS:
                    sims_dict[lang1][lang2][param] = get_model_dist(lang1_vec, lang2_vec,
                                                                    dist_func=cos_dist, param=param)
        np.save(out_dir, sims_dict)