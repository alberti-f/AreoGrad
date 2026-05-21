# bootstrap PLS regression to estimate confidence intervals for loadings and scores

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

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
n_bootstraps = SMK.params["n_bootstraps"]

area_path = SMK.input["area_path"]
gradient_path = SMK.input["gradient_path"]
dispersion_path = SMK.input["dispersion_path"]
model_path = SMK.input["model_path"]
out_path = SMK.output[0]

nsamples = dataset.N

def align_components(reference, components):
    if reference.shape[1] != components.shape[1]:
        raise ValueError("Number of components must match for alignment.")
    if reference.ndim ==1: reference = reference.reshape(-1, 1)
    if components.ndim ==1: components = components.reshape(-1, 1)

    corr = np.corrcoef(reference.T, components.T)
    corr = corr[:reference.shape[1], reference.shape[1]:]
    if reference.shape[1] > 1:
        cost = np.abs(corr)
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
        components = components[:, col_ind]
        flip = np.sign(corr[row_ind, col_ind])
    elif reference.shape[1] == 1:
        col_ind = np.array([0])
        flip = np.sign(corr).squeeze()
    components *= flip

    return components, col_ind, flip

final_model = vu.load_hdf5(model_path)
surf_area = np.load(area_path)
dispersion = np.atleast_2d(np.load(dispersion_path)[:, :n_gradients])

X = surf_area
y = dispersion
if y.ndim == 1: y = y.reshape(-1,1)
cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

random_states = np.arange(random_state, random_state + n_bootstraps)
boot_idx = np.array([
    resample(np.arange(nsamples), n_samples=nsamples, random_state=rs).flatten()
    for rs in random_states
])
x_wghts_boot = np.zeros((n_bootstraps, X.shape[1], final_model["n_components"]))
y_wghts_boot = np.zeros((n_bootstraps, y.shape[1], final_model["n_components"]))
x_scores_boot = np.zeros((n_bootstraps, X.shape[0], final_model["n_components"]))
y_scores_boot = np.zeros((n_bootstraps, y.shape[0], final_model["n_components"]))


for i in range(n_bootstraps):
    idx = boot_idx[i]
    X_boot = StandardScaler().fit_transform(X[idx].copy())
    y_boot = y[idx].copy()
    model_boot = PLSRegression(scale=False, n_components=final_model["n_components"])
    model_boot.fit(X_boot, y_boot)

    x_wghts_boot[i], idx, flp = align_components(final_model["x_weights_"], model_boot.x_weights_)
    y_wghts_boot[i] = model_boot.y_weights_[:, idx] * flp
    x_scores_boot[i] = model_boot.x_scores_[:, idx] * flp
    y_scores_boot[i] = model_boot.y_scores_[:, idx] * flp

boot_dict = {
    "index_boot": boot_idx,
    "x_weights_boot": x_wghts_boot,
    "y_weights_boot": y_wghts_boot,
    "x_scores_boot": x_scores_boot,
    "y_scores_boot": y_scores_boot,
}
vu.save_hdf5(boot_dict, out_path)
