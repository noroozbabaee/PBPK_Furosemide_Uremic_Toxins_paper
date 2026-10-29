"""
parameter_audit.py
------------------
Check the active parameters before a simulation starts

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
run_sim() in simulation.py calls audit_compounds() just before the ODEs are
solved. The function compares the parameters that are about to be used
(compound dictionaries and physiology) with the reference values of the
paper and prints a table:

    PASS  value equals the reference value (within a tolerance)
    WARN  value differs from the reference value

A WARN is not an error. In a sensitivity analysis or in a scenario that
changes a parameter on purpose (for example fu_p of furosemide), WARN lines
are expected. They help you see which parameters differ from the paper.

The reference values are NOT typed in this file. They are read from
compound_parameters.py and constants.py when the module is imported, so
there is only one place where each number is defined.

Notes
-----
Because the reference values are read at import time, simulation.py reloads
this module (importlib.reload) before each audit. That way an edit of
compound_parameters.py during an interactive session is picked up.
Run this file directly (python parameter_audit.py) for a quick test.
"""

from compound_parameters import get_compound_parameters

from constants import (
    CKD_SCALE_FACTORS,     # GFR scale: {"healthy": 1.0, "ckd5": 0.10}
    CKD5_RBF_SCALE,        # renal blood flow scale in CKD-5
    CKD_URINE_SCALE,       # urine flow scale in CKD-5
    ML_MIN_TO_L_H,         # 0.06 (mL/min -> L/h)
    MW_IS, MW_HA, MW_pCS,
    _HEALTHY_TUBULAR_FLOWS_ML_MIN,
    _RENAL_BLOOD_FLOW_BASE_ML_MIN,
)


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------

def _build_registry() -> dict:
    """
    Build the table of reference values.

    Key format: "<compound>.<parameter>" or "<compound>.<parameter>@<mode>".
    A key with "@healthy" or "@ckd5" is only checked in that mode; a key
    without "@" is checked in both modes.
    Each entry has: expected value, tolerance in %, a short source note
    and the units.
    """
    drug_h, t1_h, t2_h, t3_h = get_compound_parameters("healthy")
    drug_c, t1_c, t2_c, t3_c = get_compound_parameters("ckd5")

    # Physiology values derived from constants.py
    GFR_scale_h  = CKD_SCALE_FACTORS["healthy"]
    GFR_scale_c  = CKD_SCALE_FACTORS["ckd5"]
    P1_healthy   = _HEALTHY_TUBULAR_FLOWS_ML_MIN["P1"]     # filtrate entering P1 = GFR
    CDC5_healthy = _HEALTHY_TUBULAR_FLOWS_ML_MIN["CDC5"]   # flow leaving CDC5 = urine

    Q_GFR_h    = P1_healthy * GFR_scale_h * ML_MIN_TO_L_H
    Q_GFR_c    = P1_healthy * GFR_scale_c * ML_MIN_TO_L_H
    Q_kidney_h = _RENAL_BLOOD_FLOW_BASE_ML_MIN * ML_MIN_TO_L_H
    Q_kidney_c = _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD5_RBF_SCALE * ML_MIN_TO_L_H
    Q_urine_h  = CDC5_healthy * ML_MIN_TO_L_H
    Q_urine_c  = CDC5_healthy * CKD_URINE_SCALE * ML_MIN_TO_L_H

    def entry(expected, tol_pct, source, units):
        return {"expected": expected, "tol_pct": tol_pct,
                "source": source, "units": units}

    return {
        # -- Furosemide ---------------------------------------------------------
        "drug.fu_p@healthy": entry(drug_h["fu_p"], 5.0,
            "furosemide unbound fraction, healthy (Rane 1978)", "-"),
        "drug.fu_p@ckd5": entry(drug_c["fu_p"], 40.0,
            "furosemide unbound fraction, CKD-5 (higher in uremia; Rane 1978)", "-"),
        "drug.BP": entry(drug_h["BP"], 5.0,
            "furosemide blood-to-plasma ratio", "-"),
        "drug.k_elim_hinv@healthy": entry(drug_h["k_elim_hinv"], 20.0,
            "non-renal elimination rate constant", "1/h"),
        "drug.k_elim_hinv@ckd5": entry(drug_c["k_elim_hinv"], 30.0,
            "non-renal elimination rate constant (unchanged in CKD-5)", "1/h"),
        "drug.Km_baso_uptake": entry(drug_h["Km_baso_uptake"], 5.0,
            "OAT1 Km of furosemide, 38.9 uM (Hasannejad 2004)", "mg/L"),
        "drug.IR@healthy": entry(drug_h["IR"], 1.0,
            "default infusion rate (120 mg over 2 h)", "mg/h"),
        "drug.IR@ckd5": entry(drug_c["IR"], 1.0,
            "default infusion rate (same dose in CKD-5)", "mg/h"),

        # -- Indoxyl sulfate (tox1) --------------------------------------------
        "tox1.fu_p@healthy": entry(t1_h["fu_p"], 10.0,
            "IS unbound fraction, healthy (Deltombe 2015)", "-"),
        "tox1.fu_p@ckd5": entry(t1_c["fu_p"], 15.0,
            "IS unbound fraction, CKD-5 (Deltombe 2015)", "-"),
        "tox1.BP": entry(t1_h["BP"], 5.0, "IS blood-to-plasma ratio", "-"),
        "tox1.Km_baso_uptake": entry(t1_h["Km_baso_uptake"], 5.0,
            f"IS OAT1 Km, 18 uM x MW {MW_IS}/1000 (Chang 2023)", "mg/L"),
        "tox1.Ki_furo_to_IS_mgL": entry(t1_h["Ki_furo_to_IS_mgL"], 5.0,
            "Ki furosemide, from sensitivity analysis (healthy)", "mg/L"),

        # -- Hippuric acid (tox2) ----------------------------------------------
        "tox2.fu_p@healthy": entry(t2_h["fu_p"], 10.0,
            "HA unbound fraction, healthy (Deltombe 2015)", "-"),
        "tox2.fu_p@ckd5": entry(t2_c["fu_p"], 15.0,
            "HA unbound fraction, CKD-5 (Deltombe 2015)", "-"),
        "tox2.BP": entry(t2_h["BP"], 5.0, "HA blood-to-plasma ratio", "-"),
        "tox2.Km_baso_uptake": entry(t2_h["Km_baso_uptake"], 5.0,
            f"HA OAT1 Km, 24 uM x MW {MW_HA}/1000 (Chang 2023)", "mg/L"),
        "tox2.Ki_furo_to_HA_mgL": entry(t2_h["Ki_furo_to_HA_mgL"], 5.0,
            "Ki furosemide, same value for all toxins", "mg/L"),

        # -- p-Cresyl sulfate (tox3) -------------------------------------------
        "tox3.fu_p@healthy": entry(t3_h["fu_p"], 10.0,
            "pCS unbound fraction, healthy (Deltombe 2015)", "-"),
        "tox3.fu_p@ckd5": entry(t3_c["fu_p"], 20.0,
            "pCS unbound fraction, CKD-5 (Deltombe 2015)", "-"),
        "tox3.BP": entry(t3_h["BP"], 5.0, "pCS blood-to-plasma ratio", "-"),
        "tox3.Km_baso_uptake": entry(t3_h["Km_baso_uptake"], 5.0,
            f"pCS OAT1 Km, 120 uM x MW {MW_pCS}/1000 (Chang 2023)", "mg/L"),
        "tox3.Ki_furo_to_pCS_mgL": entry(t3_h["Ki_furo_to_pCS_mgL"], 5.0,
            "Ki furosemide, same value for all toxins", "mg/L"),

        # -- Physiology ---------------------------------------------------------
        "phys.V_ven0": entry(5.0, 1.0,
            "central (blood) volume", "L"),
        "phys.Q_kidney0@healthy": entry(round(Q_kidney_h, 4), 5.0,
            "renal blood flow, healthy", "L/h"),
        "phys.Q_kidney0@ckd5": entry(round(Q_kidney_c, 4), 10.0,
            f"renal blood flow, CKD-5 (x {CKD5_RBF_SCALE})", "L/h"),
        "phys.Q_GFR@healthy": entry(round(Q_GFR_h, 4), 5.0,
            "GFR, healthy", "L/h"),
        "phys.Q_GFR@ckd5": entry(round(Q_GFR_c, 4), 10.0,
            f"GFR, CKD-5 (x {GFR_scale_c})", "L/h"),
        "phys.Q_urine@healthy": entry(round(Q_urine_h, 4), 20.0,
            "urine flow, healthy", "L/h"),
        "phys.Q_urine@ckd5": entry(round(Q_urine_c, 4), 30.0,
            f"urine flow, CKD-5 (x {CKD_URINE_SCALE})", "L/h"),
    }


# Built once when the module is imported (see the note in the header).
EXPECTED_PARAMS = _build_registry()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_compounds(
    drug: dict,
    tox1: dict,
    tox2: dict,
    tox3: dict,
    phys: dict,
    scenario: str = "healthy",
    strict: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Compare the active parameters with the reference values.

    Parameters
    ----------
    drug, tox1, tox2, tox3 : compound dictionaries (compounds.py)
    phys     : physiology dictionary (physiology.make_phys)
    scenario : "healthy" or "ckd5"
    strict   : if True, raise ValueError when a result has status "error"
               (the check only gives PASS or WARN, so no error is raised)
    verbose  : print the full table (otherwise only WARN lines)

    Returns
    -------
    dict with lists "pass", "warn" and "error"
    """
    compound_map = {
        "drug": drug, "tox1": tox1, "tox2": tox2,
        "tox3": tox3, "phys": phys,
    }
    results = {"pass": [], "warn": [], "error": []}

    if verbose:
        print()
        print("=" * 72)
        print(f"  PARAMETER CHECK  -  scenario: {scenario}")
        print("=" * 72)
        print(f"  {'Parameter':<36} {'Reference':>10} {'Actual':>10}  Status")
        print("-" * 72)

    for reg_key, spec in EXPECTED_PARAMS.items():

        # keys with "@mode" are only checked in that mode
        if "@" in reg_key:
            base, key_scenario = reg_key.rsplit("@", 1)
            if key_scenario != scenario:
                continue
        else:
            base = reg_key

        cmpd_name, param_name = base.split(".", 1)
        cmpd = compound_map.get(cmpd_name)
        if cmpd is None:
            continue

        actual   = cmpd.get(param_name, "MISSING")
        expected = spec["expected"]
        tol_pct  = spec["tol_pct"]
        units    = spec["units"]

        status  = "PASS"
        message = ""

        if actual == "MISSING":
            status  = "WARN"
            message = f"key '{param_name}' not found in {cmpd_name}"
        else:
            actual_f = float(actual)
            exp_f    = float(expected)
            abs_tol  = abs(exp_f) * tol_pct / 100.0
            if abs(actual_f - exp_f) > abs_tol:
                rel = abs(actual_f - exp_f) / max(abs(exp_f), 1e-12) * 100.0
                status  = "WARN"
                message = (f"differs by {rel:.1f}% from reference "
                           f"{exp_f:.4g} {units} ({spec['source']})")

        if verbose or status != "PASS":
            exp_s = f"{expected:.4g}"
            act_s = (f"{float(actual):.4g}" if isinstance(actual, (int, float))
                     else str(actual))
            print(f"  {status:<4} {base:<36} {exp_s:>10} {act_s:>10}  {message}")

        results[status.lower()].append(
            {"key": base, "actual": actual, "expected": expected, "message": message}
        )

    if verbose:
        print("-" * 72)
        n_p, n_w = len(results["pass"]), len(results["warn"])
        print(f"  {n_p} PASS  |  {n_w} WARN")
        print("=" * 72)
        print()

    if strict and results["error"]:
        lines = [f"  [{r['key']}] {r['message']}" for r in results["error"]]
        raise ValueError("Parameter check failed:\n" + "\n".join(lines))

    return results


# ---------------------------------------------------------------------------
# Quick test: python parameter_audit.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import physiology

    print("Test 1 - healthy reference set: expect 0 WARN")
    r = audit_compounds(*get_compound_parameters("healthy"),
                        physiology.make_phys("healthy"),
                        scenario="healthy", strict=False)
    assert len(r["warn"]) == 0, r["warn"]

    print("Test 2 - CKD-5 reference set: expect 0 WARN")
    r = audit_compounds(*get_compound_parameters("ckd5"),
                        physiology.make_phys("ckd5"),
                        scenario="ckd5", strict=False, verbose=False)
    assert len(r["warn"]) == 0, r["warn"]

    print("Test 3 - changed IS fu_p is reported as WARN")
    d, t1, t2, t3 = get_compound_parameters("healthy")
    t1["fu_p"] = 0.30
    r = audit_compounds(d, t1, t2, t3, physiology.make_phys("healthy"),
                        scenario="healthy", strict=False, verbose=False)
    assert len(r["warn"]) == 1

    print("\nAll tests passed.")
