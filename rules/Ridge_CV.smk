# Aggregate individual data into a single file for easier analysis
rule stack_data:
    input:
        gradient_paths = expand(
            f'{OUTDIR}/{{subj_id}}/{{subj_id}}.rFC_Gradients.{PARCELLATION}_{SCALE}.npy',
            subj_id=SUBJECTS
            ),
        area_paths = expand(
            f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy',
            subj_id=SUBJECTS
            )
    output:
        f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    script:
        "scripts/07.stack_data.py"

# Ridge regression with cross-validation
rule ridge_cv:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.h5",
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.h5"
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    script:
        "scripts/08.Ridge_cv.py"

# Ridge regression permutations
rule ridge_cv_perm:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5",
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5"
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    log:
        f'{OUTDIR}/logs/parcel_FC.{PARCELLATION}_{SCALE}.G{{n_gradients}}.perm{{i_perm}}.log'
    script:
        "scripts/08.Ridge_cv_perm.py"

# Permutation test
rule ridge_perm_test:
    input:
        true_path = f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.h5",
        perm_paths = lambda wc: expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5",
            n_gradients=wc.n_gradients, i_perm=range(config["n_permutations"])
        ),
        true_path_md = f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.h5",
        perm_paths_md = lambda wc: expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/permutations/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm{{i_perm}}.h5",
            n_gradients=wc.n_gradients, i_perm=range(config["n_permutations"])
        )
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm_test.csv",
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm_test.csv"
    params:
        random_state = config["random_state"]
    log:
        f'{OUTDIR}/logs/parcel_FC.{PARCELLATION}_{SCALE}.G{{n_gradients}}.perm_test.log'
    script:
        "scripts/08.Ridge_perm_test.py"

rule ablation_test:
    input: 
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/ablation_scores_G1-G{{n_gradients}}.{config['abl_metric']}.rs{config['random_state']}.npy"
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"],
        n_gradients = config["n_components"],
        abl_metric = config["abl_metric"]
    script:
        "scripts/08.ablation_test.py"