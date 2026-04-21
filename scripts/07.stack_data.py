import os
import variograd_utils as vu
import numpy as np
from sklearn.metrics import pairwise_distances

SMK = snakemake

dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

area_paths = SMK.input["area_paths"]
gradient_paths = SMK.input["gradient_paths"]
dataset = vu.dataset(dataset_id)

area_all = np.array([np.load(p) for p in area_paths])
if not os.path.exists(os.path.dirname(SMK.output["area_out"])):
    os.makedirs(os.path.dirname(SMK.output["area_out"]))
np.save(SMK.output["area_out"], area_all)

gradients_all = np.array([np.load(p) for p in gradient_paths])
if not os.path.exists(os.path.dirname(SMK.output["gradient_out"])):
    os.makedirs(os.path.dirname(SMK.output["gradient_out"]))
np.save(SMK.output["gradient_out"], gradients_all)

dispersion_all = np.array([
    [np.mean(pairwise_distances(g[:, None], metric="euclidean")) for g in subj.T]
    for subj in gradients_all
    ])
if not os.path.exists(os.path.dirname(SMK.output["dispersion_out"])):
    os.makedirs(os.path.dirname(SMK.output["dispersion_out"]))
np.save(SMK.output["dispersion_out"], dispersion_all)
