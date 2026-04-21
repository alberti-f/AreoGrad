import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

X = surf_area
X_norm = StandardScaler().fit_transform(X)
y = dispersion
if y.ndim == 1: y = y.reshape(-1,1)
cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

param_grid = {
    "n_components": np.arange(1,10)
}

model = PLSRegression(scale=False)
best_params, _, _ = cvz.grid_search(param_grid, X_norm, y.copy(),
                                    model, cv_split, metric="rmse")

final_model = PLSRegression(scale=False, **best_params)
final_model.fit(X_norm, y.copy())
out_dict = {k: val
            for k, val in final_model.__dict__.items()
            if isinstance(val, (np.ndarray, int, float, np.floating, np.integer))}
vu.save_hdf5(out_dict, out_path)
