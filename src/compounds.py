"""
compounds.py
------------
Compound parameters that are active during a simulation

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
Link between the parameter definitions and the ODE system:

    compound_parameters.py  ->  compounds.py  ->  ode_system_twocompartment.py
    (parameter values)          (active set)      (reads C.drug, C.tox1, ...)

This module
  1. gets the parameter dictionaries for "healthy" or "ckd5"
     (get_compound_parameters),
  2. builds the passive diffusion parameters per segment (DRG_PASS,
     TOX1_PASS, TOX2_PASS, TOX3_PASS) with passive_diffusion.py,
  3. makes sure each toxin has a Ki in mg/L,
  4. keeps the result in the module variables drug, tox1, tox2, tox3.

Notes
-----
The ODE reads the module variables directly. If you change a value in
compounds.drug during a session, the next simulation uses the new value.
Call initialize_compounds(mode) to go back to the original values.
"""

import numpy as np
from compound_parameters import get_compound_parameters, print_parameter_changes

# ---------------------------------------------------------------------------
# Module-level variables read by the ODE system
# ---------------------------------------------------------------------------
# The ODE does:  import compounds as C ; drug = C.drug
# They are filled by initialize_compounds() (called at the end of this file).
drug = tox1 = tox2 = tox3 = None
DRG_PASS = TOX1_PASS = TOX2_PASS = TOX3_PASS = {}


# ---------------------------------------------------------------------------
# Ki of furosemide for OAT1 uptake of the toxins
# ---------------------------------------------------------------------------
# Ki = 0.35 mg/L, selected from a per-toxin sensitivity analysis (see the
# header of compound_parameters.py). The value used by the ODEs is the one
# in compound_parameters.py; the constants below are only a fallback in case
# the mg/L key is missing from a toxin dictionary.
# 0.35 mg/L x 1000 / 330.74 g/mol = 1.058 uM
_KI_FURO_UM  = 1.058   # [uM]
_KI_FURO_MGL = 0.35    # [mg/L]


def _uM_to_mgL(uM, MW_g_mol):
    """
    Convert a concentration from umol/L to mg/L.

    Formula: C [mg/L] = C [uM] x MW [g/mol] / 1000

    Example: 1.058 uM furosemide (MW 330.74 g/mol)
             = 1.058 x 330.74 / 1000 = 0.35 mg/L
    """
    return (float(uM) * float(MW_g_mol)) / 1000.0


def _finalize_inhibition_units():
    """
    Make sure every toxin dictionary has its Ki in mg/L.

    The ODE reads Ki_furo_to_IS_mgL, Ki_furo_to_HA_mgL and
    Ki_furo_to_pCS_mgL (step 10 in kidney_rhs). If a mg/L key is missing,
    it is computed from the uM key, or else set to _KI_FURO_MGL.

    This function only prepares the parameters. Inhibition is switched on
    separately with drug["inhibit"] = True (scenario_manager.py).
    """
    global drug, tox1, tox2, tox3

    if not isinstance(drug, dict):
        return

    drug_mw = drug.get("MW", 330.74)   # furosemide MW fallback

    # (toxin_dict, mg/L key name, uM key name, fallback mg/L value)
    inhibition_targets = [
        (tox1, "Ki_furo_to_IS_mgL",  "Ki_furo_to_IS_uM",  _KI_FURO_MGL),
        (tox2, "Ki_furo_to_HA_mgL",  "Ki_furo_to_HA_uM",  _KI_FURO_MGL),
        (tox3, "Ki_furo_to_pCS_mgL", "Ki_furo_to_pCS_uM", _KI_FURO_MGL),
    ]

    for tox, mgL_key, uM_key, default_mgL in inhibition_targets:
        if not isinstance(tox, dict):
            continue

        # A value that is already set (from compound_parameters.py) is kept.
        if tox.get(mgL_key) is None:
            if tox.get(uM_key) is not None:
                # convert from uM
                tox[mgL_key] = _uM_to_mgL(tox[uM_key], drug_mw)
            else:
                # fallback value
                tox[mgL_key] = default_mgL


def initialize_compounds(mode: str = "healthy"):
    """
    Load all compound dictionaries for a physiological mode.

    Call this before a simulation (it is called once automatically, with
    mode "healthy", when the module is imported). Afterwards the module
    variables drug, tox1, tox2, tox3 and the passive-diffusion bundles
    DRG_PASS, TOX1_PASS, ... are ready for the ODE.

    Parameters
    ----------
    mode : "healthy" or "ckd5"
        Physiological mode. CKD5 applies Vmax scaling from compound_parameters.py.
    """
    global drug, tox1, tox2, tox3
    global DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS

    # Step 1: parameter values from compound_parameters.py
    drug, tox1, tox2, tox3 = get_compound_parameters(mode)

    # Step 2: furosemide is switched on by the scenario, not here
    drug.setdefault("enabled", False)

    # Step 3: passive diffusion parameters for each segment
    # (permeability x surface area clearances and the un-ionised fractions
    # alpha, beta, gamma; see passive_diffusion.py)
    from passive_diffusion import build_passive_params
    DRG_PASS  = build_passive_params("drg",  drug, mode=mode)
    TOX1_PASS = build_passive_params("tox1", tox1, mode=mode)
    TOX2_PASS = build_passive_params("tox2", tox2, mode=mode)
    TOX3_PASS = build_passive_params("tox3", tox3, mode=mode)

    # Step 4: Ki in uM, stored for information only (the ODE uses mg/L).
    # setdefault only writes the value if the key does not exist yet.
    tox1.setdefault("Ki_furo_to_IS_uM",  _KI_FURO_UM)
    tox2.setdefault("Ki_furo_to_HA_uM",  _KI_FURO_UM)
    tox3.setdefault("Ki_furo_to_pCS_uM", _KI_FURO_UM)

    # Step 5: make sure the mg/L Ki values exist
    _finalize_inhibition_units()

    # Step 6: for CKD-5, print which parameters differ from healthy
    if mode != "healthy":
        print_parameter_changes()


def load_compounds(mode: str = "healthy"):
    """
    Same as initialize_compounds(), but also returns the four dictionaries
    (drug, IS, HA, pCS).
    """
    initialize_compounds(mode)
    return drug, tox1, tox2, tox3


def apply_drug_protocol(proto: dict | None):
    """
    Change dosing settings of furosemide without reloading all compounds.
    Only the keys present in `proto` are changed.

    Example:
        apply_drug_protocol({"IR": 40.0, "t_inf_h": 1.0, "fu_p": 0.15})

    Keys that can be updated via this function:
        protocol    : "plasma_britz" or "nina"
        IR          : infusion rate [mg/h]
        t_inf_h     : infusion duration [h]
        t_start_h   : infusion start time [h]
        bolus_mg    : IV bolus dose [mg]
        k_elim_hinv : elimination rate constant [h^-1]
        fu_p        : unbound fraction in plasma
        BP          : blood/plasma ratio
    """
    global drug

    if not proto or not isinstance(drug, dict):
        return

    for key in (
        "protocol", "IR", "t_inf_h", "t_start_h", "bolus_mg",
        "k_elim_hinv", "fu_p", "BP",
    ):
        if key in proto:
            drug[key] = proto[key]


def update_compound_parameters(mode: str = "healthy"):
    """
    Reload compound parameters for a new physiological mode.
    Does the same as initialize_compounds().

    Example:
        update_compound_parameters("ckd5")
        sol_ckd = run_sim(48, mode="ckd5")
    """
    global drug, tox1, tox2, tox3
    global DRG_PASS, TOX1_PASS, TOX2_PASS, TOX3_PASS

    # Step 1: parameter values
    drug, tox1, tox2, tox3 = get_compound_parameters(mode)
    drug.setdefault("enabled", False)

    # Step 2: passive diffusion parameters
    from passive_diffusion import build_passive_params
    DRG_PASS  = build_passive_params("drg",  drug, mode=mode)
    TOX1_PASS = build_passive_params("tox1", tox1, mode=mode)
    TOX2_PASS = build_passive_params("tox2", tox2, mode=mode)
    TOX3_PASS = build_passive_params("tox3", tox3, mode=mode)

    # Step 3: Ki in uM (information only)
    tox1.setdefault("Ki_furo_to_IS_uM",  _KI_FURO_UM)
    tox2.setdefault("Ki_furo_to_HA_uM",  _KI_FURO_UM)
    tox3.setdefault("Ki_furo_to_pCS_uM", _KI_FURO_UM)

    # Step 4: make sure the mg/L Ki values exist
    _finalize_inhibition_units()

    # Step 5: for CKD-5, print the changed parameters
    if mode != "healthy":
        print_parameter_changes()


def validate_compounds():
    """
    Check that the compound dictionaries are complete.

    Raises ValueError if a compound is None or a required key is missing.
    When inhibition is on, it also checks the Ki values and prints a
    [WARN] line for suspicious values (without stopping the run).
    run_sim() calls this function before integrating.
    """
    required_keys = [
        "Km_baso_uptake", "Vmax_baso_uptake",
        "Km_api_secr",    "Vmax_api_secr",
        "CL_sec_api",     "fu_p", "BP",
        "MW",             "pKa",  "type",
    ]

    for name, comp in [
        ("Drug", drug), ("IS", tox1), ("HA", tox2), ("pCS", tox3)
    ]:
        if comp is None:
            raise ValueError(
                f"{name} compound is None - call initialize_compounds() first."
            )
        missing = [k for k in required_keys if k not in comp]
        if missing:
            raise ValueError(
                f"{name} is missing required keys: {missing}"
            )

    # -- Ki check (only when inhibition is on) --------------------------------
    if isinstance(drug, dict) and drug.get("inhibit", False):
        ki_checks = [
            (tox1, "Ki_furo_to_IS_mgL",  "IS"),
            (tox2, "Ki_furo_to_HA_mgL",  "HA"),
            (tox3, "Ki_furo_to_pCS_mgL", "pCS"),
        ]
        for tox, ki_key, tname in ki_checks:
            if not isinstance(tox, dict):
                continue
            ki_val = tox.get(ki_key, None)

            # inhibition on but no valid Ki: stop
            if ki_val is None or ki_val <= 0:
                raise ValueError(
                    f"{tname}: inhibition is active (drug['inhibit']=True) "
                    f"but {ki_key} is missing or <= 0. "
                    f"The paper value is 0.35 mg/L."
                )

            # a much lower value than 0.35 mg/L may be a unit error
            if ki_val < 0.01:
                print(
                    f"[WARN] {tname} {ki_key} = {ki_val:.4f} mg/L is very low "
                    f"(paper value 0.35 mg/L). Check the units (mg/L, not uM)."
                )

            # a very large Ki means no inhibition
            if ki_val > 1e6:
                print(
                    f"[WARN] {tname} {ki_key} = {ki_val:.2e} mg/L (effectively "
                    f"infinite). Furosemide inhibition of {tname} OAT1 is OFF."
                )

    return True


# ---------------------------------------------------------------------------
# Load the healthy parameter set when the module is imported.
# For CKD-5 call initialize_compounds("ckd5") (scenario_manager does this).
# ---------------------------------------------------------------------------
initialize_compounds("healthy")