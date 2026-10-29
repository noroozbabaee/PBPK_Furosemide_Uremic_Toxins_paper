"""
scenario_manager.py
-------------------
Simulation scenarios: healthy / CKD-5, with or without furosemide

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
set_scenario(key) prepares the compound dictionaries (C.drug, C.tox1,
C.tox2, C.tox3) for one scenario. Call it before run_sim().

Scenarios
---------
    key                     mode     furosemide   inhibition
    H_NO_DRUG               healthy  no           no
    H_WITH_DRUG_NO_INT      healthy  yes          no
    H_WITH_DRUG             healthy  yes          yes
    CKD_NO_DRUG             ckd5     no           no
    CKD_WITH_DRUG_NO_INT    ckd5     yes          no
    CKD_WITH_DRUG           ckd5     yes          yes
    CKD_HIGH_DOSE_*         ckd5     as above, with 240 mg instead of 120 mg

Default furosemide dose: 120 mg IV over 2 h (IR = 60 mg/h), in both
healthy and CKD-5 (values from compound_parameters.py).

Typical workflow
----------------
    from scenario_manager import set_scenario
    from simulation import run_sim
    from state_vector import IDX

    # 1) toxin steady state without furosemide (long run)
    set_scenario("CKD_NO_DRUG")
    sol_ss = run_sim(300.0, mode="ckd5", dt=1.0)
    ic = {name: sol_ss.y[i, -1] for name, i in IDX.items()}

    # 2) start from that steady state and give furosemide
    set_scenario("CKD_WITH_DRUG")
    sol = run_sim(12.0, mode="ckd5", custom_initial_conds=ic, dt=0.05)

Notes
-----
The ODE reads the switches C.drug["enabled"] and C.drug["inhibit"], the
unbound fraction C.drug["fu_p"] and the Ki values C.tox*["Ki_furo_to_*_mgL"].
Use the same `mode` in set_scenario() and run_sim().
"""

from __future__ import annotations

from typing import Dict, Tuple

import compounds as C


# ---------------------------------------------------------------------------
# Scenario table
# ---------------------------------------------------------------------------
# Each scenario is a tuple (mode, drug_enabled, inhibit):
#   mode         : "healthy" or "ckd5" (passed to initialize_compounds)
#   drug_enabled : True/False -> C.drug["enabled"]
#   inhibit      : True/False -> C.drug["inhibit"]
# The "_NO_INT" scenarios give furosemide without inhibition. Comparing
# "_WITH_DRUG" with "_WITH_DRUG_NO_INT" shows the effect of the inhibition
# alone.

_PRESETS: Dict[str, Tuple[str, bool, bool]] = {

    # -- Healthy scenarios -----------------------------------------------------
    "H_NO_DRUG": (
        "healthy", False, False,
        # healthy, no furosemide (toxin steady state)
    ),
    "H_WITH_DRUG_NO_INT": (
        "healthy", True, False,
        # healthy, furosemide given, inhibition OFF
    ),
    "H_WITH_DRUG": (
        "healthy", True, True,
        # healthy, furosemide given, inhibition ON
    ),

    # -- CKD-5 scenarios (default dose 120 mg over 2 h) -------------------------
    "CKD_NO_DRUG": (
        "ckd5", False, False,
        # CKD-5, no furosemide (toxin steady state)
    ),
    "CKD_WITH_DRUG_NO_INT": (
        "ckd5", True, False,
        # CKD-5, furosemide given, inhibition OFF
    ),
    "CKD_WITH_DRUG": (
        "ckd5", True, True,
        # CKD-5, furosemide given, inhibition ON
    ),

    # -- CKD-5 scenarios with a higher dose ------------------------------------
    # IR = _CKD_HIGH_DOSE_IR (120 mg/h for 2 h = 240 mg). fu_p is NOT changed
    # (stays 0.05). Use configure_ckd_dose() for other doses or fu_p values.
    "CKD_HIGH_DOSE_NO_DRUG": (
        "ckd5", False, False,
        # same as CKD_NO_DRUG
    ),
    "CKD_HIGH_DOSE_NO_INT": (
        "ckd5", True, False,
        # CKD-5, 240 mg, inhibition OFF
    ),
    "CKD_HIGH_DOSE": (
        "ckd5", True, True,
        # CKD-5, 240 mg, inhibition ON
    ),
}


# Infusion rate of the high-dose scenarios [mg/h] (2 h infusion = 240 mg).
_CKD_HIGH_DOSE_IR: float = 120.0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def set_scenario(scenario_key: str) -> None:
    """
    Load the compound parameters and switches for a named scenario.

    Parameters
    ----------
    scenario_key : str
        One of the keys defined in _PRESETS. Case-sensitive.
        Valid keys: H_NO_DRUG, H_WITH_DRUG_NO_INT, H_WITH_DRUG,
                    CKD_NO_DRUG, CKD_WITH_DRUG_NO_INT, CKD_WITH_DRUG,
                    CKD_HIGH_DOSE_NO_DRUG, CKD_HIGH_DOSE_NO_INT,
                    CKD_HIGH_DOSE

    Raises
    ------
    ValueError
        If scenario_key is not in _PRESETS.

    Side effects
    ------------
    Modifies C.drug, C.tox1, C.tox2, C.tox3 in place.
    Previous scenario settings are fully replaced.

    Example
    -------
    >>> set_scenario("H_WITH_DRUG")
    [SCENARIO] H_WITH_DRUG - mode=healthy, drug=ON, inhibit=ON
    """
    if scenario_key not in _PRESETS:
        valid = ", ".join(_PRESETS.keys())
        raise ValueError(
            f"Unknown scenario key: '{scenario_key}'. "
            f"Valid keys: {valid}"
        )

    mode, drug_enabled, inhibit = _PRESETS[scenario_key]

    # -- Step 1: load the parameters of the mode ------------------------------
    # This resets drug, tox1, tox2, tox3. It must come before the switches
    # below, otherwise the reload would remove them again.
    if hasattr(C, "initialize_compounds"):
        C.initialize_compounds(mode)
    elif hasattr(C, "load_compounds"):
        C.load_compounds(mode)
    else:
        raise RuntimeError(
            "compounds module has neither initialize_compounds() nor "
            "load_compounds(). Cannot configure scenario."
        )

    # -- Step 2: furosemide on/off ---------------------------------------------
    # With enabled = False there is no infusion, so furosemide stays at 0.
    C.drug["enabled"] = drug_enabled

    # no drug: set all dosing values to zero
    if not drug_enabled:
        C.drug["IR"]        = 0.0
        C.drug["t_start_h"] = 1e9    # start time far in the future
        C.drug["t_inf_h"]   = 0.0
        C.drug["bolus_mg"]  = 0.0

    # -- Step 3: inhibition on/off ---------------------------------------------
    # The ODE only reads C.drug["inhibit"] to decide on inhibition:
    #   True : Km_eff = Km * (1 + Cu_furo/Ki)
    #   False: Km_eff = Km
    C.drug["inhibit"] = inhibit

    # -- Step 4: high-dose scenarios -------------------------------------------
    # All other parameters come from initialize_compounds(mode) in step 1.
    # The high-dose scenarios only change the infusion rate.
    if mode == "ckd5" and drug_enabled and "HIGH_DOSE" in scenario_key:
        C.drug["IR"] = _CKD_HIGH_DOSE_IR

    # -- Step 5: print the scenario --------------------------------------------
    fu_p_str = f"fu_p={C.drug.get('fu_p', '?')}"
    ir_str   = f"IR={C.drug.get('IR', 0)} mg/h"
    dose_str = f"({C.drug.get('IR', 0) * C.drug.get('t_inf_h', 0):.0f} mg total)"
    print(
        f"[SCENARIO] {scenario_key} - "
        f"mode={mode}, "
        f"drug={'ON' if drug_enabled else 'OFF'}, "
        f"inhibit={'ON' if inhibit else 'OFF'}"
        + (f" | {fu_p_str}, {ir_str} {dose_str}" if drug_enabled else "")
    )


# ---------------------------------------------------------------------------
# Other doses (call after set_scenario)
# ---------------------------------------------------------------------------

def configure_ckd_dose(
    total_dose_mg: float,
    fu_p: float,
    t_inf_h: float = 2.0,
) -> None:
    """
    Set the furosemide dose and unbound fraction for a simulation.

    Call this AFTER set_scenario() with a scenario in which furosemide is
    enabled. It replaces IR, fu_p and t_inf_h.

    Parameters
    ----------
    total_dose_mg : float
        Total IV dose [mg]. IR = total_dose_mg / t_inf_h.

    fu_p : float
        Unbound fraction of furosemide in plasma (between 0 and 1).
        Model values: 0.02 healthy, 0.05 CKD-5.

    t_inf_h : float
        Infusion duration [h]. Default 2.0 h.

    The printed line gives a rough estimate of the peak inhibition factor
    alpha = 1 + Cu/Ki, from a simple one-compartment calculation. It is only
    a guide; the full model result comes from run_sim().

    Raises
    ------
    ValueError
        If dose <= 0, fu_p not in (0, 1), or t_inf_h <= 0.

    Example
    -------
    >>> set_scenario("CKD_WITH_DRUG")
    >>> configure_ckd_dose(80, fu_p=0.05)    # 80 mg over 2 h
    >>> sol = run_sim(t_end_h=12, mode="ckd5", custom_initial_conds=ic)
    """
    if total_dose_mg <= 0:
        raise ValueError(f"total_dose_mg must be > 0, got {total_dose_mg}")
    if not (0 < fu_p < 1):
        raise ValueError(f"fu_p must be between 0 and 1, got {fu_p}")
    if t_inf_h <= 0:
        raise ValueError(f"t_inf_h must be > 0, got {t_inf_h}")
    if C.drug is None:
        raise RuntimeError("Drug not initialised. Call set_scenario() first.")
    if not C.drug.get("enabled", False):
        raise RuntimeError(
            "Drug is not enabled in the current scenario. "
            "Call set_scenario() with a drug-enabled scenario first "
            "(e.g. 'CKD_WITH_DRUG')."
        )

    IR = total_dose_mg / t_inf_h

    # Rough estimate of the inhibition strength (printed only):
    # one compartment (Vcen = 5 L), clearance = non-renal + an assumed
    # renal clearance of 0.64 L/h, concentration at the end of the infusion.
    import math
    k_elim = float(C.drug.get("k_elim_hinv", 0.12))
    Vcen   = 5.0   # L
    BP     = float(C.drug.get("BP", 0.82))
    Ki_any = float(C.tox1.get("Ki_furo_to_IS_mgL", 0.35)) if C.tox1 else 0.35
    CL_met = k_elim * Vcen
    CLR_CKD  = 0.10 * 6.40
    CL_tot   = CL_met + CLR_CKD
    Cmax_est = (IR / CL_tot) * (1 - math.exp(-CL_tot / Vcen * t_inf_h))
    Cu_est   = Cmax_est * fu_p / BP
    alpha    = 1 + Cu_est / Ki_any
    flux_dn  = (alpha - 1) / alpha * 100

    C.drug["fu_p"]    = fu_p
    C.drug["IR"]      = IR
    C.drug["t_inf_h"] = t_inf_h

    print(
        f"[DOSE] {total_dose_mg:.0f} mg over {t_inf_h:.1f} h  "
        f"(IR={IR:.1f} mg/h) | fu_p={fu_p} | "
        f"Cmax~{Cmax_est:.1f} mg/L | Cu~{Cu_est:.3f} mg/L | "
        f"alpha~{alpha:.3f} | OAT1 flux down {flux_dn:.1f}%"
    )


def print_current_config() -> None:
    """
    Print the active furosemide and toxin settings.

    Useful after set_scenario() to check the settings before a run.
    """
    print("\n=== Current compound configuration ===")

    d = C.drug
    if d is None:
        print("  Drug: NOT INITIALISED")
    else:
        ir   = float(d.get("IR",       0.0))
        tinf = float(d.get("t_inf_h",  0.0))
        print(f"  Drug:")
        print(f"    enabled  = {d.get('enabled')}")
        print(f"    inhibit  = {d.get('inhibit')}")
        print(f"    fu_p     = {d.get('fu_p')}")
        print(f"    IR       = {ir} mg/h  ({ir * tinf:.0f} mg total)")
        print(f"    k_elim   = {d.get('k_elim_hinv')} h^-1")

    for name, tox, ki_key in [
        ("IS",  C.tox1, "Ki_furo_to_IS_mgL"),
        ("HA",  C.tox2, "Ki_furo_to_HA_mgL"),
        ("pCS", C.tox3, "Ki_furo_to_pCS_mgL"),
    ]:
        if tox is None:
            print(f"  {name}: NOT INITIALISED")
        else:
            ki = tox.get(ki_key, "not set")
            print(
                f"  {name}: enabled={tox.get('enabled', True)}, "
                f"fu_p={tox.get('fu_p')}, "
                f"Vmax_baso={tox.get('Vmax_baso_uptake')} mg/h, "
                f"Ki_furo={ki} mg/L"
            )

    print()


def reset_to_healthy_baseline() -> None:
    """
    Go back to the healthy parameter set without furosemide.
    Same as set_scenario("H_NO_DRUG").
    """
    set_scenario("H_NO_DRUG")
    print("[SCENARIO] Reset to healthy baseline - drug disabled.")


# List of all scenario names
VALID_SCENARIO_KEYS = list(_PRESETS.keys())