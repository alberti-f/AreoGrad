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
n_gradients = np.int32(SMK.wildcards["n_gradients"])

true_path = SMK.input["true_path"]
perm_paths = SMK.input["perm_paths"]

out_path = SMK.output[0]

metric = "mean_rmse"

perm_res = pd.DataFrame([vu.load_hdf5(p) for p in perm_paths])
real_res = cvz.summarize_results(vu.load_hdf5(true_path).values())
perm_test = np.mean(np.mean(perm_res[metric]) <= np.mean(real_res[metric]))
perm_res.loc["real"] = [real_res[k] for k in perm_res.columns.to_list()]
perm_res.loc["p", metric] = perm_test

perm_res.to_csv(out_path)
if os.path.exists(out_path):
    for p in perm_paths:
        os.remove(p)
