import numpy as np
import variograd_utils as vu
SMK = snakemake

dataset_id = SMK.config["dataset_id"]
dataset = vu.dataset(dataset_id)
parcellation = SMK.config["parcellation"]
parcellation_scale = SMK.config["parcellation_scale"]
subject_FCs = SMK.input

FC_avg = None
for FC_tmp in subject_FCs:
    FC_tmp = np.load(FC_tmp)
    if FC_avg is None:
        FC_avg = np.arctanh(FC_tmp)
    else:
        FC_avg += np.arctanh(FC_tmp)

FC_avg /= len(subject_FCs)
FC_avg = np.tanh(FC_avg)
print(FC_avg.shape, FC_tmp.shape)

np.save(SMK.output[0], FC_avg)