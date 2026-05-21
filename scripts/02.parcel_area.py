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
parcellation_path = SMK.params["parcellation_path"]
area_dscalar = SMK.input[0]


idx_cortex = np.hstack([hcp.vertex_info.grayl, hcp.vertex_info.grayr + hcp.vertex_info.num_meshl])

subj = vu.subject(subj_id, dataset_id)
area = nib.load(area_dscalar).get_fdata().squeeze()[idx_cortex]
labels = nib.load(parcellation_path).get_fdata().squeeze().astype(int)
if labels.size==64984:
    labels = labels[idx_cortex]
elif labels.size==59412:
    labels = labels
else:
    raise ValueError("Unexpected number of vertices in parcellation labels")

labs_start = 1 if 0 in labels else 0
area_parcs = reduce_by_labels(area, labels, red_op=np.sum)[labs_start:]

np.save(SMK.output[0], area_parcs)
