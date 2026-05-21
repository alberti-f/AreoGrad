
# script
from brainspace.gradient import GradientMaps
import numpy as np
import hcp_utils as hcp
import variograd_utils as vu
from brainspace.datasets import load_parcellation
from scipy.optimize import linear_sum_assignment
import nibabel as nib
SMK = snakemake

subj_id = SMK.wildcards["subj_id"]
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

vinfo = hcp.vertex_info
Wref = np.load(SMK.input[0])
W = np.load(SMK.input[1])
reference = np.load(SMK.input[2])

if split_hemispheres:
    labels = nib.load(parcellation_path).get_fdata().squeeze()
    cortex_l = vinfo.grayl if labels.size==64984 else np.arange(vinfo.grayl.size)
    cortex_r = vinfo.grayr + vinfo.num_meshl if labels.size==64984 else np.arange(vinfo.grayr.size) + cortex_l.size
    labels_l, labels_r = labels[cortex_l], labels[cortex_r]
    start_l = 1 if 0 in labels_l else 0
    start_r = 1 if 0 in labels_r else 0
    labels_l, labels_r = np.unique(labels_l)[start_l:], np.unique(labels_r)[start_r:]
    n_l, n_r = labels_l.size, labels_r.size

    Wref_l = Wref[:n_l, :n_l].copy()
    W_l = W[:n_l, :n_l].copy()
    gm_l = GradientMaps(n_components=n_components, approach=approach, kernel=kernel, 
                        alignment="joint", random_state=random_state)
    gm_l.fit([Wref_l, W_l], sparsity=threshold)
    gradients_l = gm_l.gradients_[1]

    Wref_r = Wref[n_l: , n_l: ].copy()
    W_r = W[n_l: , n_l: ].copy()
    gm_r = GradientMaps(n_components=n_components, approach=approach, kernel=kernel,
                        alignment="joint", random_state=random_state)
    gm_r.fit([Wref_r, W_r], sparsity=threshold)
    gradients_r = gm_r.gradients_[1]

    reference_l = reference[:n_l, :]
    similarity = np.corrcoef(gradients_l.T, reference_l.T)[:n_components, n_components:]
    row_ind, col_ind = linear_sum_assignment(np.abs(similarity), maximize=True)
    gradients_l = gradients_l[:, row_ind]
    signs = np.sign(similarity[row_ind, col_ind])
    signs[signs == 0] = 1
    gradients_l = gradients_l * signs

    reference_r = reference[n_l:, :]
    similarity = np.corrcoef(gradients_r.T, reference_r.T)[:n_components, n_components:]
    row_ind, col_ind = linear_sum_assignment(np.abs(similarity), maximize=True)
    gradients_r = gradients_r[:, row_ind]
    signs = np.sign(similarity[row_ind, col_ind])
    signs[signs == 0] = 1
    gradients_r = gradients_r * signs

    gradients = np.vstack([gradients_l, gradients_r])

else:
    gm = GradientMaps(n_components=n_components, approach=approach, kernel=kernel, 
                        alignment="joint", random_state=random_state)
    gm.fit([Wref, W], sparsity=threshold)
    gradients = gm.gradients_[1]

    similarity = np.corrcoef(gradients.T, reference.T)[:n_components, n_components:]
    row_ind, col_ind = linear_sum_assignment(np.abs(similarity), maximize=True)
    gradients = gradients[:, row_ind]
    signs = np.sign(similarity[row_ind, col_ind])
    signs[signs == 0] = 1
    gradients = gradients * signs



subj = vu.subject(subj_id, dataset_id)
np.save(SMK.output[0], gradients)
