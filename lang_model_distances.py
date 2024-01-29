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
def get_model_dist(m1, m2, dist_func, params=["all"], axis=None):
    m1_sd = m1.state_dict()
    m2_sd = m2.state_dict()
    m1_weights = []
    m2_weights = []
    param_to_vec_dists = {}
    for param in params:
        if param == "all":
            param_keys = m1_sd.keys()
        elif param == "embeddings":
            param_keys = ["shared.weight"]
        elif param == "ff_layers":
            param_keys = [p for p in list(m1_sd.keys()) if "DenseReluDense" in p]
        elif param == "attn":
            param_keys = [p for p in list(m1_sd.keys()) if "Attention" in p]
        elif param == "norm":
            param_keys = [p for p in list(m1_sd.keys()) if "norm" in p]
        else:
            param_keys = [param]
        for param_key in param_keys:
            m1_weights.append(m1_sd[param_key].detach().numpy().flatten())
            m2_weights.append(m2_sd[param_key].detach().numpy().flatten())
        vec_dist = dist_func(np.hstack(m1_weights),
                                np.hstack(m2_weights), axis=axis)
        param_to_vec_dists[param] = vec_dist
    return param_to_vec_dists


def load_model_lang_vecs(model_langs, batch_idxs):
    batch_langs_to_vecs = {}
    for idx in tqdm(batch_idxs):
        idx_lang = model_langs[idx]
        idx_path = f"{SCRIPTS_DIR}lang_models/{model_langs[idx]}"
        idx_model = AutoModelForSeq2SeqLM.from_pretrained(idx_path).eval()
        idx_vec = get_task_vector(pretrained_model, idx_model, alpha=1.0)
        batch_langs_to_vecs[idx_lang] = idx_vec
        del idx_model
    return batch_langs_to_vecs


SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"
PRETRAINED_MODEL = "google/mt5-small"
PARAMS = ["embeddings", "ff_layers", "attn"]#, "norm"]
NUM_BATCHES = 3

if __name__ == "__main__":
    out_dir = f"{SCRIPTS_DIR}model_distances/mt5-small_lang_model_cos_sims"
    #lang_model_vecs = []
    model_langs = []
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    for lang in os.listdir(f"{SCRIPTS_DIR}lang_models/"):
        if os.listdir(f"{SCRIPTS_DIR}lang_models/{lang}"):
    #        lang_model_path = f"{SCRIPTS_DIR}lang_models/{lang}"
    #        lang_model = AutoModelForSeq2SeqLM.from_pretrained(lang_model_path).eval()
    #        lang_vec = get_task_vector(pretrained_model, lang_model, alpha=1.0)
            model_langs.append(lang)
    #        lang_model_vecs.append(lang_vec)
    #        del lang_model

    sims_dict = defaultdict(dict)
    with torch.no_grad():
        batch_idxs = np.array_split(np.array(list(range(len(model_langs)))), NUM_BATCHES)
        for b1_idxs in batch_idxs:
            # Load batch 1 vectors
            print("batch 1", b1_idxs)
            batch1_lang_to_vec = load_model_lang_vecs(model_langs, b1_idxs)

            for b2_idxs in batch_idxs:
                # Load batch 2 vectors
                print("batch 2", b2_idxs)
                batch2_lang_to_vec = {}
                if set(b1_idxs) == set(b2_idxs):
                    batch2_lang_to_vec = batch1_lang_to_vec
                # if b2 idxs are before b1 idxs, then we don't need to compute their
                # vectors because all cases will fall into i > j below
                elif b2_idxs[0] > b1_idxs[-1]:
                    batch2_lang_to_vec = load_model_lang_vecs(model_langs, b2_idxs)

                # Compute distances between b1 vecs and b2 vecs
                print("Computing cos similarities between batch 1 and batch 2 vectors")
                for i in tqdm(b1_idxs):
                    for j in b2_idxs:
                        lang1 = model_langs[i]
                        lang2 = model_langs[j]
                        if i > j:
                            sims_dict[lang1][lang2] = {p: sims_dict[lang2][lang1][p] for p in PARAMS}
                        elif i == j:
                            sims_dict[lang1][lang2] = {p: 1.0 for p in PARAMS}
                        else:
                            sims_dict[lang1][lang2] = get_model_dist(batch1_lang_to_vec[lang1],
                                                                    batch2_lang_to_vec[lang2],
                                                                    dist_func=cos_dist,
                                                                    params=PARAMS)
                    np.save(out_dir, sims_dict)
                del batch2_lang_to_vec
            del batch1_lang_to_vec