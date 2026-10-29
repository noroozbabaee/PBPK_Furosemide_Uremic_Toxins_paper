"""
state_vector.py
---------------
Names, order and initial values of the model states

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
The ODE solver works with one long vector y. This module defines
  STATE              list of state names (the order of y)
  IDX                dictionary name -> position in y
  initial_conditions builds y at t = 0

Example
-------
    from state_vector import IDX
    C_IS = sol.y[IDX["C_tox1_cen"]]     # IS in the central compartment
    Ae_IS = sol.y[IDX["Ae_t1"]]         # IS excreted in urine [mg]

Units: concentrations mg/L, Ae_* (cumulative excreted amount) mg.
"""

import numpy as np
from constants import SEGMENTS

# ---------------------------------------------------------------------------
# Names of all states, in the order used in the vector y
# ---------------------------------------------------------------------------
# Systemic states and cumulative urinary amounts first:
STATE = [
    "C_ven",
    "C_peri",
    "C_tox1_cen",
    "C_tox2_cen",
    "C_tox3_cen",
    "C_bladder",
    "Ae_drg",
    "Ae_t1",
    "Ae_t2",
    "Ae_t3"
]

# then 12 states per segment (4 compounds x vascular/lumen/cell):
#   C_<seg>_V, C_<seg>_T, C_<seg>_C           furosemide
#   C_tox1_<seg>_V, ... C_tox3_<seg>_C        IS, HA, pCS
# 10 + 11 segments x 12 = 142 states in total.
for seg in SEGMENTS:
    STATE += [f"C_{seg}_V", f"C_{seg}_T", f"C_{seg}_C"]
    STATE += [f"C_tox1_{seg}_V", f"C_tox1_{seg}_T", f"C_tox1_{seg}_C"]
    STATE += [f"C_tox2_{seg}_V", f"C_tox2_{seg}_T", f"C_tox2_{seg}_C"]
    STATE += [f"C_tox3_{seg}_V", f"C_tox3_{seg}_T", f"C_tox3_{seg}_C"]

# IDX["C_ven"] gives the position of a state in y (and in sol.y)
IDX = {name: i for i, name in enumerate(STATE)}


def initial_conditions(
        phys: dict,
        custom_initial_conds: dict | None = None,
        *,
        custom_initial_conditions: dict | None = None,
        compounds: dict | None = None,
        scenario: dict | None = None,
        prev_end: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the initial state vector y0.

    Priority:
      1. prev_end (a full state vector) is copied as it is.
      2. values in custom_initial_conds are used for the states they name.
      3. other states: furosemide 0 (or bolus/V_ven0), toxins C0 from the
         compound dictionaries, kidney compartments 0 or see below.
    """

    n_states = len(IDX)
    y0 = np.zeros(n_states, dtype=float)

    # both argument names are accepted
    init_dict = custom_initial_conds if custom_initial_conds is not None else custom_initial_conditions

    # full state vector given: use it
    if prev_end is not None:
        if len(prev_end) != n_states:
            raise ValueError(f"prev_end length {len(prev_end)} != number of states {n_states}")
        y0[:] = prev_end
        return y0

    # values given by name
    if init_dict:
        for name, val in init_dict.items():
            if name in IDX:
                y0[IDX[name]] = float(val)
            else:
                print(f"Warning: Unknown state '{name}' in custom initial conditions")

    # peripheral compartment of furosemide starts empty
    if "C_peri" in IDX and (init_dict is None or "C_peri" not in init_dict):
        y0[IDX["C_peri"]] = 0.0

    # active compound dictionaries
    try:
        import compounds as C
        drug = C.drug
        tox1 = C.tox1
        tox2 = C.tox2
        tox3 = C.tox3
    except (ImportError, AttributeError):
        # compounds not available
        drug = {}
        tox1 = tox2 = tox3 = {}

    # furosemide IV bolus (if used): C_ven(0) = bolus / V_ven0
    if "C_ven" in IDX and (init_dict is None or "C_ven" not in init_dict):
        bolus = float(drug.get("bolus_mg", 0.0))
        drug_enabled = bool(drug.get("enabled", False))

        if drug_enabled and bolus > 0.0:
            V_ven0 = float(phys.get("V_ven0", 5.0))
            y0[IDX["C_ven"]] = bolus / V_ven0
            print(f"[IC] furosemide bolus = {bolus} mg, C_ven = {y0[IDX['C_ven']]:.6f} mg/L")

    # toxins: start at C0
    for tox_name, tox_dict in [("C_tox1_cen", tox1), ("C_tox2_cen", tox2), ("C_tox3_cen", tox3)]:
        if tox_name in IDX and (init_dict is None or tox_name not in init_dict):
            y0[IDX[tox_name]] = float(tox_dict.get("C0", 0.0))

    # -- Peritubular (vascular) compartments at the start ----------------------
    # Fresh start only (no custom values given). If the vascular segments
    # started at 0 while the central concentration is > 0, the renal
    # exchange term (Qkid/Vcen)*(C_ren_exit - C_cen) would remove compound
    # from the central compartment very quickly at t = 0 (time constant
    # Vcen/Qkid = 5/72 h, about 4 min). To avoid this artificial start-up
    # effect, every vascular segment starts at the concentration that leaves
    # the glomerulus:
    #     C_in = C_cen * (1 - (GFR/Qkid) * fu_p/BP)
    if init_dict is None:
        GFR   = float(phys.get("Q_GFR",      7.2))
        Qkid  = float(phys.get("Q_kidney0",  72.0))
        EPS   = 1e-12

        # list of (state prefix, unused, C_in)
        _cmpd_vascular = []

        # Drug
        C_ven_val = float(y0[IDX.get("C_ven", 0)])
        if C_ven_val > 0.0:
            fu_p_d = float(drug.get("fu_p", 0.02))
            BP_d   = float(drug.get("BP",   0.82))
            C_in_d = C_ven_val * (1.0 - (GFR / max(Qkid, EPS)) * (fu_p_d / max(BP_d, EPS)))
            _cmpd_vascular.append(("C", None, C_in_d))

        # Toxins
        for prefix, tox_dict, cen_key in (
            ("C_tox1", tox1, "C_tox1_cen"),
            ("C_tox2", tox2, "C_tox2_cen"),
            ("C_tox3", tox3, "C_tox3_cen"),
        ):
            C_cen = float(y0[IDX.get(cen_key, 0)])
            if C_cen > 0.0:
                fu_p_t = float(tox_dict.get("fu_p", 0.04))
                BP_t   = float(tox_dict.get("BP",   0.70))
                C_in_t = C_cen * (1.0 - (GFR / max(Qkid, EPS)) * (fu_p_t / max(BP_t, EPS)))
                C_in_t = max(C_in_t, 0.0)
                _cmpd_vascular.append((prefix, None, C_in_t))

        # same C_in in every segment (a simple starting value; the solver
        # then finds the real profile along the nephron)
        for seg in SEGMENTS:
            # furosemide
            if len(_cmpd_vascular) > 0:
                C_in_d = _cmpd_vascular[0][2] if _cmpd_vascular[0][0] == "C" else 0.0
                vkey_d = f"C_{seg}_V"
                if vkey_d in IDX and C_in_d > 0.0:
                    y0[IDX[vkey_d]] = C_in_d

            # toxins
            for prefix, _, C_in_t in _cmpd_vascular:
                if prefix.startswith("C_tox"):
                    vkey_t = f"{prefix}_{seg}_V"
                    if vkey_t in IDX and C_in_t > 0.0:
                        y0[IDX[vkey_t]] = C_in_t

        if len(_cmpd_vascular) > 0:
            print(
                f"[IC] vascular segments start at the post-glomerular concentration."
            )
            for prefix, _, C_in in _cmpd_vascular:
                print(f"  {prefix}: C_in = {C_in:.4e} mg/L  (all segments)")

    return y0


def get_initial_conditions_dict(mode: str = "healthy") -> dict:
    """Default initial values (systemic states only) as a dictionary."""
    from compounds import get_compound_parameters
    drug_params, tox1_params, tox2_params, tox3_params = get_compound_parameters(mode)

    return {
        "C_ven": 0.0,
        "C_peri": 0.0,
        "C_tox1_cen": tox1_params["C0"],
        "C_tox2_cen": tox2_params["C0"],
        "C_tox3_cen": tox3_params["C0"],
        "C_bladder": 0.0,
        "Ae_drg": 0.0,
        "Ae_t1": 0.0,
        "Ae_t2": 0.0,
        "Ae_t3": 0.0
    }
