import os
import variograd_utils as vu
import numpy as np

SMK = snakemake

dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

area_paths = SMK.input["area_paths"]
gradient_paths = SMK.input["gradient_paths"]
dataset = vu.dataset(dataset_id)

area_all = np.array([np.load(p) for p in area_paths])
outpath = vu.dataset(dataset_id).outpath(f"AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{parcellation}_{parcellation_scale}.npy")
if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))
np.save(outpath, area_all)

gradients_all = np.array([np.load(p) for p in gradient_paths])
outpath = vu.dataset(dataset_id).outpath(f"AreaResults/All.rFC_Gradients.{parcellation}_{parcellation_scale}.npy")
if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))
np.save(outpath, gradients_all)
