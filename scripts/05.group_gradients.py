
# script
from brainspace.gradient import GradientMaps
import numpy as np
import hcp_utils as hcp
import variograd_utils as vu
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

labels_l, labels_r = load_parcellation(parcellation, scale=parcellation_scale, join=False)
labels_l, labels_r = np.unique(labels_l[hcp.vertex_info.grayl])[1:], np.unique(labels_r[hcp.vertex_info.grayr])[1:]
n_l, n_r = labels_l.size, labels_r.size

W = np.load(SMK.input[0])

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
dataset = vu.dataset(dataset_id)
np.save(SMK.output[0], gradients)
