# Compute vertex area
rule vertex_area:
    output:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii'
    params:
        lh_surf = config["surfaces"]["lh"],
        rh_surf = config["surfaces"]["rh"]
    log:
        f'{OUTDIR}/logs/vertex_area.{PARCELLATION}_{SCALE}.{{subj_id}}.log'
    script:
        f"{SCRIPTS}/01.vertex_area.py"

# Compute parcel area
rule parcel_area:
    input:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii',
    output:
        f'{OUTDIR}/{{subj_id}}/{{subj_id}}.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy'
    params:
        parcellation_path = PARCELLATION_PATH
    log:
        f'{OUTDIR}/logs/parcel_area.{PARCELLATION}_{SCALE}.{{subj_id}}.log'
    script:
        f"{SCRIPTS}/02.parcel_area.py"