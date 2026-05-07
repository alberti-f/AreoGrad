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

runs = [f"{subj.dir}/{run}" for run in SMK.params.values()]

tseries = []
for run in runs:
    tseries32k = nib.load(run).get_fdata()[:, :n_cortex]
    ts_parc = reduce_by_labels(tseries32k, labels)[:, labs_start:]
    ts_parc = zscore(ts_parc, axis=0)
    tseries.append(ts_parc)

tseries_parcs = np.vstack(tseries).T
fc_rest = np.corrcoef(tseries_parcs)

np.save(SMK.output["ts"], tseries_parcs)
np.save(SMK.output["fc"], fc_rest)
