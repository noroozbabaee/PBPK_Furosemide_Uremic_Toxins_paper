"""
physiology.py
-------------
Physiological parameters (volumes and flows) for healthy and CKD-5

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
make_phys(mode) returns the dictionary `phys` that the ODE uses:
central volume, renal blood flow, GFR, urine flow and the tubular flows.
flows_L_per_h(phys) returns the tubular flows as a list (FLOWS_L_HR).

Tubular fluid path:
    Bowman's space -> P1 -> P2 -> P3 -> LH1 -> LH2 -> DT
                   -> CDC1 -> ... -> CDC5 -> urine

GFR and renal blood flow are scaled separately in CKD-5
(GFR x 0.10, renal blood flow x 0.35, see constants.py).

Notes
-----
The numbers themselves are defined in constants.py; this module only
combines them. Change values there, not here.
"""

from typing import Dict, List, Optional

from constants import (
    PER_KIDNEY,
    SEGMENTS,
    get_tubular_flows_ml_min,
    validate_flows,
    get_renal_blood_flow_ml_min,
    CKD_SCALE_FACTORS,
    CKD5_RBF_SCALE,
    ML_MIN_TO_L_H,
)
import constants as _K




def _mode_scale(mode: str) -> float:
    """Return the GFR/tubular-flow CKD scale factor for a given mode."""
    return float(CKD_SCALE_FACTORS.get(mode, 0.10))


def _build_mode_flows(
    mode: str,
    custom_scale: Optional[float] = None,
) -> tuple[Dict[str, float], float, float]:
    """
    Tubular flows and renal blood flow for the requested mode.
    GFR (and the flows P1 ... DT) and renal blood flow are scaled with
    separate factors (constants.CKD_SCALE_FACTORS and CKD5_RBF_SCALE).

    Returns
    -------
    tubular_flows : dict
        Tubular segment flows in mL/min.

    renal_blood_flow_ml_min : float
        Total renal blood flow in mL/min, both kidneys.

    effective_gfr_scale : float
        GFR/tubular-flow scale used for the selected mode.
    """
    # start from the healthy values
    healthy_tubular_flows = get_tubular_flows_ml_min("healthy")
    healthy_rbf_ml_min = get_renal_blood_flow_ml_min("healthy")

    if mode == "healthy":
        effective_gfr_scale = 1.0
        effective_rbf_scale = 1.0

    elif mode == "ckd5":
        effective_gfr_scale = _mode_scale("ckd5")
        effective_rbf_scale = CKD5_RBF_SCALE

    else:
        raise ValueError(f"mode must be 'healthy' or 'ckd5', got '{mode}'")

    # Optional (sensitivity analysis): custom_scale replaces the GFR scale.
    # It changes the flows P1 ... DT only, not renal blood flow and not the
    # collecting duct flows.
    if custom_scale is not None:
        effective_gfr_scale = float(custom_scale)

    # flows of the mode (the CKD-5 collecting duct flows are set in
    # constants._rebuild_mode_maps)
    tubular_flows = get_tubular_flows_ml_min(mode)

    # custom_scale: rescale P1 ... DT only
    if custom_scale is not None:
        from constants import PROX, NONPROX
        for seg in PROX + NONPROX:
            tubular_flows[seg] = healthy_tubular_flows[seg] * effective_gfr_scale

    renal_blood_flow_ml_min = healthy_rbf_ml_min * effective_rbf_scale

    return tubular_flows, renal_blood_flow_ml_min, effective_gfr_scale


def make_phys(
    mode: str = "healthy",
    custom_scale: Optional[float] = None,
) -> Dict[str, object]:
    """
    Physiology dictionary for the given mode (used by the ODE as `phys`).

    Parameters
    ----------
    mode : {"healthy", "ckd5"}
        Physiological mode.

    custom_scale : float or None
        Optional GFR scale relative to healthy (sensitivity analysis only).
        Changes GFR and the flows P1 ... DT; renal blood flow is not changed.

    Returns
    -------
    dict
        Keys:

        V_ven0
            Central compartment volume [L].

        Vb_L
            Mixing volume [L] (only used by the "nina" infusion option).

        Qc0
            Cardiac output [L/h] (only used by the "nina" infusion option).

        Q_kidney0
            Total renal blood flow [L/h], both kidneys.

        Q_GFR
            Glomerular filtration rate [L/h], both kidneys.

        Q_urine
            Urine production rate [L/h].

        V_bladder
            Bladder volume [L] (only used for C_bladder).

        tubular_flows_ml_min
            Tubular segment flows [mL/min].

        mode
            Mode string.

        scale
            Effective GFR/tubular-flow scale.
    """
    if mode not in {"healthy", "ckd5"}:
        raise ValueError(f"mode must be 'healthy' or 'ckd5', got '{mode}'")

    # Step 1: flows of the mode
    tubular_flows, renal_blood_flow_ml_min, effective_gfr_scale = _build_mode_flows(
        mode=mode,
        custom_scale=custom_scale,
    )

    # Step 2: GFR = flow entering P1 ; urine flow = flow in CDC5
    #   healthy: GFR 120 mL/min, urine 1.0 mL/min
    #   CKD-5  : GFR  12 mL/min, urine 0.5 mL/min
    gfr_ml_min   = tubular_flows["P1"]
    urine_ml_min = tubular_flows["CDC5"]

    # Step 3: mL/min -> L/h (1 mL/min = 0.06 L/h)
    Q_kidney0 = renal_blood_flow_ml_min * ML_MIN_TO_L_H
    Q_GFR     = gfr_ml_min   * ML_MIN_TO_L_H
    Q_urine   = urine_ml_min * ML_MIN_TO_L_H

    if _K.VERBOSE:
        print(f"Q_kidney0 = {Q_kidney0:.4f} L/h")
        print(f"Q_GFR = {Q_GFR:.4f} L/h")
        print(f"Q_urine = {Q_urine:.4f} L/h")

    # Step 4: check the flows. A failed check prints a warning but does not
    # stop the simulation. In CKD-5 only P1 = GFR is checked here (the full
    # check is in constants.py, test 4-5).
    if mode == "healthy":
        if not validate_flows(tubular_flows, gfr_ml_min, urine_ml_min):
            print(
                f"[WARN] physiology.py: Flow validation failed for mode='{mode}'. "
                "Simulation will continue, but check constants.py flow values."
            )
    else:
        if abs(tubular_flows["P1"] / max(gfr_ml_min, 1e-12) - 1.0) > 0.1:
            print(f"[WARN] CKD P1 flow ({tubular_flows['P1']:.2f}) != GFR ({gfr_ml_min:.2f})")
        elif _K.VERBOSE:
            print("Flow validation passed!")

    # print the CKD-5 values
    if mode == "ckd5" and _K.VERBOSE:
        print(
            "[INFO] CKD5 physiology: "
            f"GFR_scale={effective_gfr_scale:.3f}, "
            f"RBF_scale={CKD5_RBF_SCALE:.3f}, "
            f"Q_GFR={Q_GFR:.3f} L/h, "
            f"Q_kidney0={Q_kidney0:.3f} L/h, "
            f"Q_urine={Q_urine:.4f} L/h ({urine_ml_min:.2f} mL/min)"
        )

    # Step 5: physiology dictionary
    return {
        # Central (blood) compartment volume, shared by furosemide and the
        # toxins. Used in the infusion term (IR/V_ven0), the renal exchange
        # term (Q_kidney0/V_ven0) and the toxin production term (R0/V_ven0).
        # For furosemide: Vss = V_ven0 * (1 + k12/k21) = 5.33 L.
        "V_ven0": 5.0,  # [L]

        # Only used by the "nina" infusion option (mixing delay).
        "Vb_L": 5.0,    # [L]
        "Qc0": 330.0,   # [L/h] cardiac output (5.5 L/min)

        # Kidney flows       healthy     CKD-5
        #   Q_kidney0        72 L/h      25.2 L/h
        #   Q_GFR            7.2 L/h     0.72 L/h
        #   Q_urine          0.06 L/h    0.03 L/h
        "Q_kidney0": Q_kidney0,  # [L/h] total renal blood flow
        "Q_GFR":     Q_GFR,     # [L/h] glomerular filtration rate
        "Q_urine":   Q_urine,   # [L/h] urine production rate

        # Bladder volume, only used for the output C_bladder
        "V_bladder": 0.6,  # [L]

        # Tubular flows [mL/min]; flows_L_per_h() converts them to L/h
        "tubular_flows_ml_min": tubular_flows,

        # information
        "mode": mode,
        "scale": effective_gfr_scale,
    }


def flows_L_per_h(phys_params: Dict[str, object]) -> List[float]:
    """
    Tubular flows as a list in SEGMENTS order [L/h].

    The ODE receives this list as FLOWS_L_HR. Entry k is the flow entering
    the lumen of segment k. The flow leaving CDC5 is the urine flow
    (phys["Q_urine"]), because the list has no entry after CDC5.

    Parameters
    ----------
    phys_params : dict
        Output of make_phys(). Must contain "tubular_flows_ml_min".

    Returns
    -------
    list of float
        Length = len(SEGMENTS). Units are L/h.
    """
    tubular_flows = phys_params["tubular_flows_ml_min"]

    # PER_KIDNEY = True: one kidney, flows x 0.5
    per_kid = 0.5 if PER_KIDNEY else 1.0
    ml_min = {segment: flow * per_kid for segment, flow in tubular_flows.items()}


    return [ml_min[segment] * ML_MIN_TO_L_H for segment in SEGMENTS]