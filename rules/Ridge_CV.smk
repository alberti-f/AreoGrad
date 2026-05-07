# Ridge regression with cross-validation
rule ridge_cv:
    input:
        gradient_path = f"{OUTDIR}/{OUTSUBDIR}/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/{OUTSUBDIR}/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.h5",
        f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.h5"
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    script:
        f"{SCRIPTS}/08.Ridge_cv.py"

# Ridge regression permutations
rule ridge_cv_perm:
    input:
        gradient_path = f"{OUTDIR}/{OUTSUBDIR}/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/{OUTSUBDIR}/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5",
        f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5"
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    log:
        f'{OUTDIR}/logs/parcel_FC.{PARCELLATION}_{SCALE}.G{{n_gradients}}.perm{{i_perm}}.log'
    script:
        f"{SCRIPTS}/08.Ridge_cv_perm.py"

# Permutation test
rule ridge_perm_test:
    input:
        true_path = f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.h5",
        perm_paths = lambda wc: expand(
            f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5",
            n_gradients=wc.n_gradients, i_perm=range(config["n_permutations"])
        ),
        true_path_md = f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.h5",
        perm_paths_md = lambda wc: expand(
            f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5",
            n_gradients=wc.n_gradients, i_perm=range(config["n_permutations"])
        )
    output:
        f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm_test.csv",
        f"{OUTDIR}/{OUTSUBDIR}/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm_test.csv"
    params:
        random_state = config["random_state"]
    log:
        f'{OUTDIR}/logs/parcel_FC.{PARCELLATION}_{SCALE}.G{{n_gradients}}.perm_test.log'
    script:
        f"{SCRIPTS}/08.Ridge_perm_test.py"