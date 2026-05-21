
# script
from brainspace.gradient import GradientMaps
import numpy as np
import hcp_utils as hcp
import variograd_utils as vu
import nibabel as nib
from brainspace.datasets import load_parcellation
SMK = snakemake

dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

n_components = SMK.params["n_components"]
threshold = SMK.params["threshold"]
approach = SMK.params["approach"]
kernel = SMK.params["kernel"]
random_state = SMK.params["random_state"]
parcellation_path = SMK.params["parcellation_path"]
split_hemispheres = SMK.params["split_hemispheres"]

W = np.load(SMK.input[0])

if split_hemispheres:
    vinfo = hcp.vertex_info
    labels = nib.load(parcellation_path).get_fdata().squeeze()
    cortex_l = vinfo.grayl if labels.size==64984 else np.arange(vinfo.grayl.size)
    cortex_r = vinfo.grayr + vinfo.num_meshl if labels.size==64984 else np.arange(vinfo.grayr.size) + cortex_l.size
    labels_l, labels_r = labels[cortex_l], labels[cortex_r]
    start_l = 1 if 0 in labels_l else 0
    start_r = 1 if 0 in labels_r else 0
    labels_l, labels_r = np.unique(labels_l)[start_l:], np.unique(labels_r)[start_r:]
    n_l, n_r = labels_l.size, labels_r.size


    W_l = W[:n_l, :n_l].copy()
    gm_l = GradientMaps(n_components=n_components, approach=approach, 
                        kernel=kernel, random_state=random_state)
    gm_l.fit(W_l, sparsity=threshold)
    gradients_l = gm_l.gradients_ 

    W_r = W[n_l: , n_l: ].copy()
    gm_r = GradientMaps(n_components=n_components, approach=approach,
                        kernel=kernel,random_state=random_state)
    gm_r.fit(W_r, sparsity=threshold)
    gradients_r = gm_r.gradients_

    gradients = np.vstack([gradients_l, gradients_r])

else:
    gm = GradientMaps(n_components=n_components, approach=approach, kernel=kernel, random_state=random_state)
    gm.fit(W, sparsity=threshold)
    gradients = gm.gradients_

dataset = vu.dataset(dataset_id)
np.save(SMK.output[0], gradients)
