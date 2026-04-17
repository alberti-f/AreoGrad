# final model for interpretation

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import pairwise_distances
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
model_path = SMK.input["model_path"]
out_path = SMK.output[0]

nsamples = dataset.N

def align_components(reference, components):
    corr = np.corrcoef(reference.T, components.T)
    corr = corr[reference.shape[1]:, :reference.shape[1]]
    cost = np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    flip = np.sign(corr[row_ind, col_ind])
    return components[:, col_ind] * flip

final_model = vu.load_hdf5(model_path)
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
cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

x_loads_boot = np.zeros((n_bootstraps, X.shape[1], final_model["n_components"]))
y_loads_boot = np.zeros((n_bootstraps, y.shape[1], final_model["n_components"]))

for i in range(n_bootstraps):
    idx = resample(np.arange(nsamples).reshape(-1, 1), n_samples=nsamples)
    idx = idx.flatten()

    X_boot = StandardScaler().fit_transform(X[idx].copy())
    y_boot = StandardScaler().fit_transform(y[idx].copy())
    model_boot = PLSRegression(scale=False, n_components=final_model["n_components"])
    model_boot.fit(X_boot, y_boot)

    x_loads_boot[i] = align_components(final_model["x_loadings_"], model_boot.x_loadings_)
    y_loads_boot[i] = align_components(final_model["y_loadings_"], model_boot.y_loadings_)

boot_dict = {
    "x_loadings_boot": x_loads_boot,
    "y_loadings_boot": y_loads_boot,
}
vu.save_hdf5(boot_dict, out_path)
