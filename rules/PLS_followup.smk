# Follow up analyses to interpret PLS results

# Final PLS model
rule pls_final_model:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy",
        dispersion_path = f"{OUTDIR}/AreaResults/All.rFC_Dispersion.{PARCELLATION}_{SCALE}.npy"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/FinalModel_G1-G{{n_gradients}}.rs{config['random_state']}.h5",
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"]
    script:
        f"{SCRIPTS}/10.PLS_final_model.py"

# PLS bootstraps for final model
rule pls_bootstraps:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy",
        dispersion_path = f"{OUTDIR}/AreaResults/All.rFC_Dispersion.{PARCELLATION}_{SCALE}.npy",
        model_path = f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/FinalModel_G1-G{{n_gradients}}.rs{config['random_state']}.h5"
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/Bootstraps_G1-G{{n_gradients}}.rs{config['random_state']}.h5"
    params:
        random_state = config["random_state"],
        n_splits = config["n_splits"],
        n_bootstraps = config["n_bootstraps"]
    script:
        f"{SCRIPTS}/10.PLS_bootstraps.py"

# Correlation between y scores of the final model and FC between networks
rule nw_fc_correlation:
    input:
        gradient_path = f"{OUTDIR}/AreaResults/All.rFC_Gradients.{PARCELLATION}_{SCALE}.npy",
        area_path = f"{OUTDIR}/AreaResults/All.T1w.midthickness_MSMAll_va.32k_fs_LR.{PARCELLATION}_{SCALE}.npy",
        dispersion_path = f"{OUTDIR}/AreaResults/All.rFC_Dispersion.{PARCELLATION}_{SCALE}.npy",
        final_model_path = f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/FinalModel_G1-G{{n_gradients}}.rs{config['random_state']}.h5",
        parcellation_path = config["parcellation_path"].format(SCALE),
        fc_paths = expand(
            f"{OUTDIR}/{{subj}}/{{subj}}.rFC_all_runs.{PARCELLATION}_{SCALE}.npy", subj=SUBJECTS
            )
    output:
        f"{OUTDIR}/AreaResults/{PARCELLATION}_{SCALE}/PLS/NW_FC_Correlation_G1-G{{n_gradients}}.rs{config['random_state']}.csv",
    params:
        random_state = config["random_state"],
        n_bootstraps = config["n_bootstraps"]
    script:
        f"{SCRIPTS}/10.PLS_FC_correlation.py"
