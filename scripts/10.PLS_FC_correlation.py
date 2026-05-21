import numpy as np
import hcp_utils as hcp
import nibabel as nib
import pandas as pd

import variograd_utils as vu


SMK = snakemake

dataset_id = SMK.config["dataset_id"]
dataset = vu.dataset(dataset_id)
parcellation = SMK.config["parcellation"]
parcellation_path = SMK.params["parcellation_path"]

random_state = SMK.params["random_state"]
n_bootstraps = SMK.params["n_bootstraps"]
n_gradients = np.int32(SMK.wildcards["n_gradients"])

area_path = SMK.input["area_path"]
gradient_path = SMK.input["gradient_path"]
dispersion_path = SMK.input["dispersion_path"]
final_model_path = SMK.input["final_model_path"]
fc_paths = SMK.input["fc_paths"]
out_path = SMK.output[0]


surf_area = np.load(area_path)
gradients = np.atleast_2d(np.load(gradient_path)[:,:,:n_gradients])
triu_idx = np.triu_indices(gradients.shape[1], k=1)
dispersion = np.load(dispersion_path)

cortex = np.hstack([hcp.vertex_info.grayl, hcp.vertex_info.grayr+hcp.vertex_info.num_meshl])

parcs = nib.load(parcellation_path)
label_map = parcs.get_fdata()[:, cortex].squeeze()
labels = np.unique(label_map)[1:]
label_df = pd.DataFrame(parcs.header.get_axis(0).get_element(0)[1]).T
label_df = label_df.loc[labels]
label_df["NW"] = label_df[0].str.split("_").str[2]
label_df["H"] = label_df[0].str.split("_").str[1]
nw_names = label_df["NW"].unique()

M = np.vstack([label_df["NW"].values == nw for nw in nw_names]).astype(np.float32)
M /= M.sum(axis=1, keepdims=True)

nw_triu_idx = np.triu_indices(len(nw_names), k=0)
nw_pairs = list(zip(nw_names[nw_triu_idx[0]], nw_names[nw_triu_idx[1]]))

fc_all = np.full([len(fc_paths), len(nw_pairs)], np.nan)
for i, p in enumerate(fc_paths):
    fc = np.load(p) 
    np.fill_diagonal(fc, 0)
    fc = np.arctanh(fc)
    fc_nw = M @ fc @ M.T
    fc_all[i, :] = np.tanh(fc_nw[nw_triu_idx])

final_model = vu.load_hdf5(final_model_path)
scores = final_model["y_scores_"]
disp_fc_corr = np.corrcoef(scores.T, fc_all.T)[:scores.shape[1], scores.shape[1]:]


a = (0.05 / disp_fc_corr.size) / 2
margins = [a, 1-a]

disp_fc_corr_bs = np.full([n_bootstraps, scores.shape[1], fc_all.shape[1]], np.nan)
for i in range(n_bootstraps):
    idx = np.random.choice(scores.shape[0], size=scores.shape[0], replace=True)
    disp_fc_corr_bs[i] = np.corrcoef(scores[idx].T, fc_all[idx].T)[:scores.shape[1], scores.shape[1]:]
disp_fc_corr_bs_z = np.arctanh(disp_fc_corr_bs)
corr_ci_low, corr_ci_high = np.percentile((disp_fc_corr_bs_z), margins, axis=0)
corr_ci_low, corr_ci_high = np.tanh(corr_ci_low), np.tanh(corr_ci_high)
corr_significant = (corr_ci_low * corr_ci_high) > 0

df = pd.DataFrame({
    "R": disp_fc_corr.flatten(),
    "NW Pair": np.hstack([f"{nw[0]}-{nw[1]}" for nw in nw_pairs] * scores.shape[1]),
    "LV": np.hstack([np.repeat(range(scores.shape[1]), len(nw_pairs)) for _ in range(1)])+1,
    "CI_low": corr_ci_low.flatten(),
    "CI_high": corr_ci_high.flatten(),
    "H1": (corr_ci_low * corr_ci_high > 0).flatten().astype(int)
    })

df.to_csv(out_path, index=False)