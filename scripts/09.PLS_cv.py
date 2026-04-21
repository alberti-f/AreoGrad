import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

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
dispersion_path = SMK.input["dispersion_path"]
out_path = SMK.output[0]

def vip(x_scores, x_weights, y_loadings):

    p, h = x_weights.shape
    vips = np.zeros((p,))
    s = np.diag(x_scores.T @ x_scores @ y_loadings.T @ y_loadings).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (x_weights[i,j] / np.linalg.norm(x_weights[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s).squeeze()
    return vips


surf_area = np.load(area_path)
dispersion = np.atleast_2d(np.load(dispersion_path)[:, :n_gradients])

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
for i, fold in enumerate(out):
   out[i]["vip"] = vip(fold["x_scores_"], fold["x_weights_"], fold["y_loadings_"])
out_dict = dict(zip(np.arange(1, n_splits+1).astype("str"), out))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
vu.save_hdf5(out_dict, out_path)
