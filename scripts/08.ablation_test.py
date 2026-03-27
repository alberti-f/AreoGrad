import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import CVz.CVz as cvz
import variograd_utils as vu
import nibabel as nib
import pandas as pd
from brainspace.datasets import load_parcellation
import hcp_utils as hcp
from tqdm import tqdm

SMK = snakemake

dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = pSMK.config["parcellation_scale"]

random_state = SMK.params["random_state"]
n_splits = SMK.params["n_splits"]
n_gradients = np.int32(SMK.wildcards["n_gradients"]) - 1

area_path = SMK.input["area_path"]
gradient_path = SMK.input["gradient_path"]

scores_diff_path = SMK.output[0]


dataset = vu.dataset(dataset_id)
surf_area = np.load(area_path)
gradients = np.load(gradient_path)

X = surf_area
C = surf_area.sum(axis=1, keepdims=True)
X = np.hstack([C, X])
n_features = X.shape[1]
param_grid = {
    "alpha": np.linspace(1, 5000, 50)
}

cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
hp_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state+1)
cv_args = {
    "y": y,
    "model": Ridge(),
    "split": cv_split,
    "inner_split": hp_split,
    "param_grid": param_grid,
    "grid_search_kw": {"metric": "r2"},
}

metric = "r2"

# Cumulative dispersion
y = np.std(gradients[:,:,:n_gradients], axis=1).sum(axis=1)
ablation_scores = np.full((n_features-1,  n_splits), np.nan)
for i in tqdm(range(n_features-1)):
    cv_args["X"] = np.delete(X, i+1, axis=1)

    out = cvz.run_cv(**cv_args)
    scores = [o[metric] for o in out]
    ablation_scores[i] = scores

CV_path = vu.dataset(dataset_id).outpath(
    f"AreaResults/{parcellation}_{parcellation_scale}/Ridge/CV_results_G1-G{{0}}_rs{random_state}.h5"
    )
full_scores = vu.load_hdf5(CV_path.format(n_gradients)).values()
full_scores = np.array([o["r2"] for o in full_scores])

scores_diff = ablation_scores - full_scores

np.save(scores_diff_path.format(n_gradients), scores_diff)


# Single gradient dispersion
y = np.std(gradients[:,:,n_gradients], axis=1)
ablation_scores = np.full((n_features-1,  n_splits), np.nan)
for i in tqdm(range(n_features-1)):
    cv_args["X"] = np.delete(X, i+1, axis=1)

    out = cvz.run_cv(**cv_args)
    scores = [o[metric] for o in out]
    ablation_scores[i] = scores

CV_path = vu.dataset(dataset_id).outpath(
    f"AreaResults/{parcellation}_{parcellation_scale}/Ridge/CV_results_G1-G{{0}}_rs{random_state}.h5"
    )
full_scores = vu.load_hdf5(CV_path.format(n_gradients)).values()
full_scores = np.array([o["r2"] for o in full_scores])

scores_diff = ablation_scores - full_scores

np.save(scores_diff_path.format(n_gradients), scores_diff)