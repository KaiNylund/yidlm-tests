import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM
from lang_model_distances import cos_dist


SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"
PRETRAINED_MODEL = "google/mt5-small"

def get_embedding_sims(m1, m2, sim_func):
    sims = []
    m1_sd = m1.state_dict()
    m2_sd = m2.state_dict()
    m1_embs = m1_sd["shared.weight"].detach().numpy()
    m2_embs = m2_sd["shared.weight"].detach().numpy()
    for i in range(len(m2_embs)):
        sims.append(sim_func(m1_embs[i], m2_embs[i]))
    return np.array(sims)


if __name__ == "__main__":
    out_dir = f"{SCRIPTS_DIR}mt5-small_lang_embedding_changes"
    model_langs = []
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    for lang in os.listdir(f"{SCRIPTS_DIR}lang_models/"):
        if os.listdir(f"{SCRIPTS_DIR}lang_models/{lang}"):
            model_langs.append(lang)

    embedding_dists = {}
    he_model = AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/he").eval()
    yi_model = AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/yi").eval()
    embedding_dists["he_to_yi"] = get_embedding_sims(he_model, yi_model, cos_dist)
    del he_model
    del yi_model

    for lang in tqdm(model_langs):
        lang_model = AutoModelForSeq2SeqLM.from_pretrained(f"{SCRIPTS_DIR}lang_models/{lang}").eval()
        embedding_dists[lang] = get_embedding_sims(pretrained_model, lang_model, cos_dist)

    np.save(out_dir, embedding_dists)
