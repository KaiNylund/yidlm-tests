import numpy as np
import umap
import os
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoModelForSeq2SeqLM
from collections import defaultdict
from tqdm import tqdm
from helper_scripts.get_task_vector import get_task_vector


def get_model_flattened_weights(model):
    model_sd = model.state_dict()
    model_weights = []
    with torch.no_grad():
        for param_name in model_sd.keys():
            model_weights.append(model_sd[param_name].detach().numpy().flatten())

        return np.hstack(np.array(model_weights))


# Returns a dict of param_name -> list of vec param projection in same order as proj_model_paths
# reduces using the given reducer
def project_vec_params(base_model_name, proj_model_paths, params, reducer, embedding_pct=0.1):
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).eval()
    vec_proj_param_weights = defaultdict(list)

    if "shared.weight" in params:
        print(f"Sampling {embedding_pct * 100}% of weights for projection")
        num_embeddings = pretrained_model.state_dict()["shared.weight"].size()[0]
        sampled_emb_idxs = np.random.choice(num_embeddings,
                                        int(num_embeddings * embedding_pct),
                                            replace=False)
    # Build dict of all params to project
    for model_path in tqdm(proj_model_paths):
        if "KaiNylund" not in model_path and not os.path.exists(model_path):
            print("missing " + model_path)
            continue
        
        print(model_path)
        proj_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
        proj_vec = get_task_vector(pretrained_model, proj_model, alpha=1.0)
        #for name, param in month_vec.named_parameters():
        #    print(name, param.size())
        model_params = dict(proj_vec.named_parameters())
        #proj_weights = np.concatenate((model_params[param_name1].detach().numpy().flatten(),
        #                               model_params[param_name2].detach().numpy().flatten()))
        for param in params:
            if param == "all":
                param_weights = get_model_flattened_weights(proj_vec)
            else:
                param_weights = model_params[param].detach().numpy()
            if param == "shared.weight":
                param_weights = param_weights[sampled_emb_idxs, :]
            vec_proj_param_weights[param].append(param_weights.flatten())
        del proj_model
        del model_params
        del proj_vec


    # Actually do all the projecting with the given reducer
    vec_param_projections = {}
    for param, vec_weights in tqdm(vec_proj_param_weights.items()):
        vec_weights = np.array(vec_weights)
        print(vec_weights.shape)
        vec_param_projections[param] = reducer.fit_transform(vec_weights)
    return vec_param_projections


PROJ_PARAMS = [
    "shared.weight",
    "decoder.block.7.layer.2.DenseReluDense.wi_0.weight",
    "decoder.block.7.layer.2.DenseReluDense.wi_1.weight",
    "decoder.block.7.layer.2.DenseReluDense.wo.weight",
    "encoder.block.7.layer.1.DenseReluDense.wi_0.weight",
    "encoder.block.7.layer.1.DenseReluDense.wi_1.weight",
    "encoder.block.7.layer.1.DenseReluDense.wo.weight",
    "decoder.block.7.layer.1.EncDecAttention.q.weight",
    "decoder.block.7.layer.1.EncDecAttention.k.weight",
    "decoder.block.7.layer.1.EncDecAttention.v.weight",
    "decoder.block.7.layer.1.EncDecAttention.o.weight",
    "encoder.block.7.layer.0.SelfAttention.q.weight",
    "encoder.block.7.layer.0.SelfAttention.k.weight",
    "encoder.block.7.layer.0.SelfAttention.v.weight",
    "encoder.block.7.layer.0.SelfAttention.o.weight"
]
PRETRAINED_MODEL = "google/mt5-small"
SCRIPTS_DIR = "/mmfs1/gscratch/ark/knylund/yidlm-tests/"


if __name__ == "__main__":
    umap_reducer = umap.UMAP(n_neighbors=15, metric="cosine")
    #tsne_reducer = TSNE(n_components=2, learning_rate='auto', init='random')
    #pca_reducer = PCA(n_components=2)

    model_langs = []
    proj_model_paths = []
    for lang in os.listdir(f"{SCRIPTS_DIR}lang_models/"):
        if os.listdir(f"{SCRIPTS_DIR}lang_models/{lang}"):
            proj_model_paths.append(f"{SCRIPTS_DIR}lang_models/{lang}")
            model_langs.append(lang)

    out_dir = f"{SCRIPTS_DIR}mt5-small_lang_model_projections"
    model_projections = project_vec_params(PRETRAINED_MODEL, proj_model_paths,
                                           PROJ_PARAMS, umap_reducer)
    model_projections["model_langs"] = model_langs
    np.save(out_dir, model_projections)