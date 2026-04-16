import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import pairwise_distances

import CVz.CVz as cvz
import variograd_utils as vu

SMK = snakemake

dataset_id = SMK.config["dataset_id"]
dataset = vu.dataset(dataset_id)
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

random_state = SMK.params["random_state"]
n_splits = SMK.params["n_splits"]
n_gradients = np.int32(SMK.wildcards["n_gradients"])

area_path = SMK.input["area_path"]
gradient_path = SMK.input["gradient_path"]
out_path = SMK.output[0]


surf_area = np.load(area_path)
gradients = np.atleast_2d(np.load(gradient_path)[:,:,:n_gradients])
triu_idx = np.triu_indices(gradients.shape[1], k=1)
dispersion = np.array([
    [np.mean(pairwise_distances(g[:, None], metric="euclidean")) for g in subj.T]
    for subj in gradients
    ])

X = surf_area
y = dispersion
if y.ndim == 1: y = y.reshape(-1,1)

param_grid = {
    "n_components": np.arange(1,10)
}

cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
hp_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state+1)

cv_args = {
    "X": X,
    "y": y,
    "model": PLSRegression(scale=False),
    "split": cv_split,
    "inner_split": hp_split,
    "param_grid": param_grid,
    "grid_search_kw": {"metric":"rmse"},
}

out = cvz.run_cv(**cv_args)
out_dict = dict(zip(np.arange(1, n_splits+1).astype("str"), out))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
vu.save_hdf5(out_dict, out_path)
