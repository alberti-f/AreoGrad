# Compute parcel area

import variograd_utils as vu
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.datasets import load_parcellation
import nibabel as nib
import numpy as np
import hcp_utils as hcp
SMK = snakemake

subj_id = SMK.wildcards["subj_id"]
dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

area_dscalar = SMK.input[0]

idx_cortex = np.hstack([hcp.vertex_info.grayl, hcp.vertex_info.grayr + hcp.vertex_info.num_meshl])

subj = vu.subject(subj_id, dataset_id)
area = nib.load(area_dscalar).get_fdata().squeeze()[idx_cortex]
labels = load_parcellation(parcellation, scale=parcellation_scale, join=True)[idx_cortex]
area_parcs = reduce_by_labels(area, labels, red_op=np.sum)[1:]

np.save(subj.outpath(f"{subj_id}.T1w.midthickness_MSMAll_va.32k_fs_LR.{parcellation}_{parcellation_scale}.npy"), area_parcs)
