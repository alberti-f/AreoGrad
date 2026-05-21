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
parcellation_path = SMK.params["parcellation_path"]
parcellate_ts = SMK.config["parcellate_ts"]

subj = vu.subject(subj_id, dataset_id)
if parcellate_ts:
    n_cortex = hcp.vertex_info.grayl.size + hcp.vertex_info.grayr.size
    idx_cortex = np.hstack([hcp.vertex_info.grayl, hcp.vertex_info.grayr + hcp.vertex_info.num_meshl])
    labels = nib.load(parcellation_path).get_fdata().squeeze().astype(int)
    if labels.size==64984:
        labels = labels[idx_cortex]
    elif labels.size==59412:
        labels = labels
    else:
        raise ValueError("Unexpected number of vertices in parcellation labels")
    labs_start = 1 if 0 in labels else 0

runs = [f"{subj.dir}/{run}" for run in SMK.params["runs"]]

tseries = []
for run in runs:
    if parcellate_ts:
        tseries32k = nib.load(run).get_fdata()[:, :n_cortex]
        ts_parc = reduce_by_labels(tseries32k, labels)[:, labs_start:]
        ts_parc = zscore(ts_parc, axis=0)
    else:
        ts_parc = zscore(np.load(run), axis=0)
    tseries.append(ts_parc)

tseries_parcs = np.vstack(tseries).T
fc_rest = np.corrcoef(tseries_parcs)

np.save(SMK.output["ts"], tseries_parcs)
np.save(SMK.output["fc"], fc_rest)
