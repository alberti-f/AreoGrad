import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import CVz.CVz as cvz
import variograd_utils as vu

SMK = snakemake

dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

random_state = SMK.params["random_state"]
n_splits = SMK.params["n_splits"]
n_gradients = np.int32(SMK.wildcards["n_gradients"]) - 1

area_path = SMK.input["area_path"]
gradient_path = SMK.input["gradient_path"]


dataset = vu.dataset(dataset_id)
surf_area = np.load(area_path)
gradients = np.load(gradient_path)
rng = np.random.default_rng()

X = surf_area
y = np.std(gradients[:,:,n_gradients], axis=1)
y = rng.permutation(y)
C = surf_area.sum(axis=1, keepdims=True)
X = np.hstack([C, X])

param_grid = {
    "alpha": np.linspace(1, 5000, 50)
}

cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
hp_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state+1)

cv_args = {
    "X": X,
    "y": y,
    "model": Ridge(),
    "split": cv_split,
    "inner_split": hp_split,
    "param_grid": param_grid,
    "grid_search_kw": {"metric": "r2"},
}

out = cvz.run_cv(**cv_args)
summary = cvz.summarize_results(out)
summary = {k: summary[k] for k in ["mean_r", "mean_r2", "mean_mae", "mean_rmse"]}

CV_path = SMK.output[0]
if not os.path.exists(os.path.dirname(CV_path)):
    os.makedirs(os.path.dirname(CV_path))
vu.save_hdf5(summary, CV_path)


y_md = y_md = np.std(gradients[:,:,:n_gradients+1], axis=1).sum(axis=1)
cv_args["y"] = rng.permutation(y_md)
out = cvz.run_cv(**cv_args)
summary = cvz.summarize_results(out)
summary = {k: summary[k] for k in ["mean_r", "mean_r2", "mean_mae", "mean_rmse"]}

CV_path = SMK.output[1]
if not os.path.exists(os.path.dirname(CV_path)):
    os.makedirs(os.path.dirname(CV_path))
vu.save_hdf5(summary, CV_path)

