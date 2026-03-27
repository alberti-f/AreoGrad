# Snakefile

import variograd_utils as vu
configfile: "config.yaml"

DATASET = config["dataset_id"]
OUTDIR = vu.dataset(DATASET).output_dir
SUBJECTS = vu.dataset(DATASET).subj_list
PARCELLATION = config["parcellation"]
SCALE = config["parcellation_scale"]

# End result
rule all:
    input:
        expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G{{n_gradients}}_rs{config['random_state']}.perm_test.csv",
            n_gradients=config["n_gradients"]
            ),
        expand(
            f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/Ridge/CV_results_G1-G{{n_gradients}}_rs{config['random_state']}.perm_test.csv",
            n_gradients=config["n_gradients"]
            )

# Compute vertex area
rule vertex_area:
    output:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii',
    log:
        f'{OUTDIR}/logs/vertex_area.{PARCELLATION}_{SCALE}.{{subj_id}}.log'
    script:
        "scripts/01.vertex_area.py"

# Compute parcel area
rule parcel_area:
    input:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii',
    output:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy',
    log:
        f'{OUTDIR}/logs/parcel_area.{PARCELLATION}_{SCALE}.{{subj_id}}.log'
    script:
        "scripts/02.parcel_area.py"

# Parcellate rs timeseries and compute FC matrices
rule parcel_FC:
    output:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.rfMRI_REST_all_runs.{PARCELLATION}_{SCALE}.npy',
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy',
    log:
        f'{OUTDIR}/logs/parcel_FC.{PARCELLATION}_{SCALE}.{{subj_id}}.log'
    script:
        "scripts/03.parcel_FC.py"

# Compute group-average FC matrix
rule group_FC:
    input:
        expand(
            f"{OUTDIR}/{{subj_id}}/{{subj_id}}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy",
            subj_id=SUBJECTS
        )
    output:
        f'{OUTDIR}/{DATASET}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy'
    script:
        "scripts/04.group_FC.py"

# Group-level gradients
rule group_gradients:
    input:
        f'{OUTDIR}/{DATASET}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy'
    output:
        f'{OUTDIR}/{DATASET}.rFC_Gradients.{PARCELLATION}_{SCALE}.npy'
    params:
        n_components = config["n_components"],
        threshold = config["threshold"],
        approach = config["approach"],
        kernel = config["kernel"],
        random_state = config["random_state"]
    script:
        "scripts/05.group_gradients.py"

# Compute aligned gradients
rule individual_gradients:
    input:
        f'{OUTDIR}/{DATASET}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy',
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy',
        f'{OUTDIR}/{DATASET}.rFC_Gradients.{PARCELLATION}_{SCALE}.npy'
    output:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.rFC_Gradients.{PARCELLATION}_{SCALE}.npy',
    params:
        n_components = config["n_components"],
        threshold = config["threshold"],
        approach = config["approach"],
        kernel = config["kernel"],
        random_state = config["random_state"]
    script:
        "scripts/06.individual_gradients.py"

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

