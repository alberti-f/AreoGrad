import os
import numpy as np
import pandas as pd
import CVz.CVz as cvz
import variograd_utils as vu

SMK = snakemake

dataset_id = SMK.config["dataset_id"]
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]

random_state = SMK.params["random_state"]
n_gradients = np.int32(SMK.wildcards["n_gradients"]) - 1

true_path = SMK.input["true_path"]
perm_paths = SMK.input["perm_paths"]
true_paths_md = SMK.input["true_path_md"]
perm_paths_md = SMK.input["perm_paths_md"]

metric = "mean_r2"


perm_res = pd.DataFrame([vu.load_hdf5(p) for p in perm_paths])
real_res = cvz.summarize_results(vu.load_hdf5(true_path).values())
perm_test = np.mean(perm_res[metric] >= real_res[metric])
perm_res.loc["real"] = [real_res[k] for k in perm_res.columns.to_list()]
perm_res.loc["p", metric] = perm_test

perm_out = SMK.output[0]
perm_res.to_csv(perm_out)
if os.path.exists(perm_out):
    for p in perm_paths:
        os.remove(p)


perm_res_md = pd.DataFrame([vu.load_hdf5(p) for p in perm_paths_md])
real_res_md = cvz.summarize_results(vu.load_hdf5(true_path).values())
perm_test_md = np.mean(perm_res_md[metric] >= real_res_md[metric])
perm_res_md.loc["real"] = [real_res_md[k] for k in perm_res_md.columns.to_list()]
perm_res_md.loc["p", metric] = perm_test_md

perm_out = SMK.output[1]
perm_res_md.to_csv(perm_out)
if os.path.exists(perm_out):
    for p in perm_paths_md:
        os.remove(p)
