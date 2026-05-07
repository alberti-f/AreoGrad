# Compute vertex area

import variograd_utils as vu
import nibabel as nib
from subprocess import run
import os
SMK = snakemake

subj_id = SMK.wildcards["subj_id"]
dataset_id = SMK.config["dataset_id"]

subj = vu.subject(subj_id, dataset_id)
if not os.path.exists(subj.outpath("")):
    os.makedirs(subj.outpath(""))

calculate_vertex_area = "wb_command -surface-vertex-areas {surface} {metric}"
create_cifti = "wb_command -cifti-create-dense-scalar {dscalar} -left-metric {lmetric} -right-metric {rmetric}"

# Left hemisphere
surface = getattr(subj, f"L_midthickness_32k_T1w")
metric_l = subj.outpath(f"{subj_id}.L.T1w.midthickness_MSMAll_va.32k_fs_LR.shape.gii")
run(calculate_vertex_area.format(surface=surface, metric=metric_l), shell=True)

# Right hemisphere
surface = getattr(subj, f"R_midthickness_32k_T1w")
metric_r = subj.outpath(f"{subj_id}.R.T1w.midthickness_MSMAll_va.32k_fs_LR.shape.gii")
run(calculate_vertex_area.format(surface=surface, metric=metric_r), shell=True)

# Create CIFTI
dscalar = SMK.output[0]
run(create_cifti.format(dscalar=dscalar, lmetric=metric_l, rmetric=metric_r), shell=True)

if os.path.exists(dscalar):
    os.remove(metric_l)
    os.remove(metric_r)
