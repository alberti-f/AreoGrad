# PLS regression with cross-validation
rule pls_cv:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/CV_results_G1-G{{n_gradients}}.rs{config['random_state']}.h5",
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    script:
        f"{SCRIPTS}/09.PLS_cv.py"

# PLS regression permutations
rule pls_cv_perm:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/permutations/CV_results_G1-G{{n_gradients}}_perm{{i_perm}}.rs{config['random_state']}.h5",
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    script:
        f"{SCRIPTS}/09.PLS_cv_perm.py"

# Permutation test
rule pls_perm_test:
    input:
        true_path = f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/CV_results_G1-G{{n_gradients}}.rs{config['random_state']}.h5",
        perm_paths = lambda wc: expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/permutations/CV_results_G1-G{{n_gradients}}_perm{{i_perm}}.rs{config['random_state']}.h5",
            n_gradients=wc.n_gradients, i_perm=range(config["n_permutations"])
        )
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/CV_results_G1-G{{n_gradients}}.rs{config['random_state']}.perm_test.csv"
    params:
        random_state = config["random_state"]
    script:
        f"{SCRIPTS}/09.PLS_perm_test.py"

