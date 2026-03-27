# Generate FC matrices

import variograd_utils as vu
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.datasets import load_parcellation
import nibabel as nib
import numpy as np
from itertools import product
import hcp_utils as hcp
from scipy.stats import zscore
SMK = snakemake

subj_id = SMK.wildcards["subj_id"]
dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

subj = vu.subject(subj_id, dataset_id)
n_cortex = hcp.vertex_info.grayl.size + hcp.vertex_info.grayr.size
idx_cortex = np.hstack([hcp.vertex_info.grayl, hcp.vertex_info.grayr + hcp.vertex_info.num_meshl])
labels = load_parcellation(parcellation, scale=parcellation_scale, join=True)[idx_cortex]
labs_start = 1 if 0 in labels else 0

runs = product([1, 2], ["LR", "RL"])
tseries_path = f"{subj.dir}/MNINonLinear/Results/rfMRI_REST{{run}}/rfMRI_REST{{run}}_Atlas_MSMAll_hp2000_clean.dtseries.nii"

tseries = []
for run, lr in runs:
    tseries32k = nib.load(tseries_path.format(run=f"{run}_{lr}")).get_fdata()[:, :n_cortex]
    ts_parc = reduce_by_labels(tseries32k, labels)[:, labs_start:]
    ts_parc = zscore(ts_parc, axis=0)
    tseries.append(ts_parc)

tseries_parcs = np.vstack(tseries).T
fc_rest = np.corrcoef(tseries_parcs)

np.save(subj.outpath(f"{subj_id}.rfMRI_REST_all_runs.{parcellation}_{parcellation_scale}.npy"), tseries_parcs)
np.save(subj.outpath(f"{subj_id}.rFC_all_runs.{parcellation}_{parcellation_scale}.npy"), fc_rest)