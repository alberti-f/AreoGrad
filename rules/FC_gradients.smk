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