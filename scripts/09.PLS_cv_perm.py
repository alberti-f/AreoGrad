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


surf_area = np.load(area_path)
dispersion = np.atleast_2d(np.load(dispersion_path)[:, :n_gradients])

rng = np.random.default_rng()
X = surf_area
y = rng.permutation(dispersion) 
y = (y - y.mean(axis=0)) / y.std(axis=0) # normalize to account for different scales of the gradients

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
summary = cvz.summarize_results(out)
out_dict = {k: summary[k] for k in ["mean_r", "mean_r2", "mean_mae", "mean_rmse"]}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
vu.save_hdf5(out_dict, out_path)
