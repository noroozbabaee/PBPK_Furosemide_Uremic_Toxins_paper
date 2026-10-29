"""
constants.py
------------
Geometry, flows, pH and numerical settings of the kidney model

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
All values that do NOT belong to a single compound:
  1. switches (USE_APICAL_MM, PER_KIDNEY, VERBOSE)
  2. segment names and order (PROX, NONPROX, CDS, SEGMENTS) and apical
     membrane areas
  3. tubular flows, healthy
  4. CKD-5 scale factors for GFR, urine flow and renal blood flow
  5. flow maps for both modes
  6. segment volumes
  7. pH values (lumen, cell, plasma)
  8. molecular weights
  9. proximal tubule geometry (segment weights, basolateral/apical area)
 10. solver tolerances and unit conversions
 11. a check of the flow maps
Compound parameters (fu_p, Km, Vmax, Ki, R0, ...) are in
compound_parameters.py.

Summary of the physiology
-------------------------
                      healthy        CKD-5
    GFR               120 mL/min     12 mL/min   (x 0.10)
    renal blood flow  1200 mL/min    420 mL/min  (x 0.35)
    urine flow        1.0 mL/min     0.5 mL/min

Notes
-----
Other modules use "import constants as K" and read K.VERBOSE etc. at the
moment they need it. Changing K.VERBOSE = False in your script therefore
works everywhere.
Run this file directly (python constants.py) for a quick test of the
flow maps.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Switches
# ---------------------------------------------------------------------------
MODE              = "healthy"   # mode used for the pH maps at import
USE_APICAL_MM     = True        # apical secretion: Michaelis-Menten (True)
                                # or linear clearance CL_sec_api (False)
PER_KIDNEY        = False       # False = both kidneys (default);
                                # True = one kidney (areas, volumes, flows x 0.5)
VERBOSE           = True        # True = print flow maps, checks and
                                # intermediate values; set False for long runs


# ---------------------------------------------------------------------------
# 2. Segments

PROX    = ["P1", "P2", "P3"]               # proximal tubule
NONPROX = ["LH1", "LH2", "DT"]            # loop of Henle + distal tubule
CDS     = ["CDC1", "CDC2", "CDC3", "CDC4", "CDC5"]  # collecting duct chain

# Order in which tubular fluid flows: Bowman's space -> urine
SEGMENTS = PROX + NONPROX + CDS           # length 11

_twoK_to_oneK = 0.5 if PER_KIDNEY else 1.0

# Apical (luminal) membrane area per segment [dm^2], both kidneys.
# The proximal tubule has by far the largest area; the brush border is
# included separately with MICROVILLI_FACTOR_APICAL (section 9).
SA_APICAL_DM2 = {
    "P1":   6107.0 * _twoK_to_oneK,   # proximal S1: brush-border dominant
    "P2":   6107.0 * _twoK_to_oneK,   # proximal S2
    "P3":   6107.0 * _twoK_to_oneK,   # proximal S3 (pars recta)
    "LH1":    61.0 * _twoK_to_oneK,   # thin descending limb
    "LH2":    61.0 * _twoK_to_oneK,   # thin/thick ascending limb
    "DT":    156.0 * _twoK_to_oneK,   # distal tubule
    "CDC1":    6.7 * _twoK_to_oneK,   # collecting duct segments 1-5
    "CDC2":    6.7 * _twoK_to_oneK,
    "CDC3":    6.7 * _twoK_to_oneK,
    "CDC4":    6.7 * _twoK_to_oneK,
    "CDC5":    6.7 * _twoK_to_oneK,
}


# ---------------------------------------------------------------------------
# 3. Tubular flows, healthy [mL/min]
# ---------------------------------------------------------------------------
# Flow entering each segment. It decreases along the nephron because water
# is reabsorbed.
#   P1   = GFR   = 120 mL/min (7.2 L/h)
#   CDC5 = urine =   1 mL/min (0.06 L/h, 1.44 L/day)

_HEALTHY_TUBULAR_FLOWS_ML_MIN = {
    "P1":  120,   # = GFR
    "P2":   94,
    "P3":   68,
    "LH1":  50,
    "LH2":  50,   # ascending limb is impermeable to water: no change
    "DT":   20,
    "CDC1": 15,
    "CDC2":  8,
    "CDC3":  4,
    "CDC4":  2,
    "CDC5":  1,   # = urine flow
}


# ---------------------------------------------------------------------------
# 4. CKD stage 5
# ---------------------------------------------------------------------------
# GFR and the flows of P1 ... DT: 10 % of healthy
# (GFR 12 mL/min; CKD stage 5 is GFR < 15 mL/min, KDIGO 2012)
CKD_SCALE_FACTORS = {"healthy": 1.0, "ckd5": 0.10}

# Urine flow is scaled separately (model assumption): patients with CKD-5
# who are not on dialysis usually still produce urine. With the GFR scale
# (0.10) the urine flow would be 0.1 mL/min (144 mL/day); with 0.50 it is
# 0.5 mL/min (720 mL/day). Used to set the CDC5 flow in _rebuild_mode_maps.
CKD_URINE_SCALE = 0.50

# Renal blood flow in CKD-5: 35 % of healthy (model assumption)
# 1200 mL/min x 0.35 = 420 mL/min = 25.2 L/h. physiology.py uses this value.
CKD5_RBF_SCALE = 0.35

# Healthy renal blood flow, both kidneys: 1200 mL/min = 72 L/h
_RENAL_BLOOD_FLOW_BASE_ML_MIN = 1200.0


# ---------------------------------------------------------------------------
# 5. Flow maps per mode
# ---------------------------------------------------------------------------

def _scaled_flows(base: dict, scale: float) -> dict:
    """Return a copy of base flows multiplied by scale."""
    return {seg: base[seg] * scale for seg in base}


def _rebuild_mode_maps() -> None:
    """
    Build _TUBULAR_FLOWS_ML_MIN and RENAL_BLOOD_FLOW_ML_MIN for both modes.

    CKD-5 flows [mL/min]:
      P1=12, P2=9.4, P3=6.8, LH1=5, LH2=5, DT=2 (healthy x 0.10)
      CDC1=2.0, CDC2=1.6, CDC3=1.2, CDC4=0.9, CDC5=0.5 (set directly)
    The collecting duct values are set directly so that the flow never
    increases from one segment to the next (CDC1 = DT) and the urine flow
    is 0.5 mL/min (0.030 L/h).
    """
    global _TUBULAR_FLOWS_ML_MIN, RENAL_BLOOD_FLOW_ML_MIN

    _TUBULAR_FLOWS_ML_MIN = {
        "healthy": _scaled_flows(
            _HEALTHY_TUBULAR_FLOWS_ML_MIN, CKD_SCALE_FACTORS["healthy"]
        ),

        "ckd5": {
            # P1 ... DT: healthy x 0.10
            **{seg: _HEALTHY_TUBULAR_FLOWS_ML_MIN[seg] * CKD_SCALE_FACTORS["ckd5"]
               for seg in PROX + NONPROX},

            # collecting duct: set directly
            "CDC1": 2.0,   # = DT in CKD-5
            "CDC2": 1.6,
            "CDC3": 1.2,
            "CDC4": 0.9,
            "CDC5": 0.5,   # -> Q_urine = 0.030 L/h
        },
    }

    RENAL_BLOOD_FLOW_ML_MIN = {
        "healthy": _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD_SCALE_FACTORS["healthy"],

        # Information only: physiology.py takes the healthy value and
        # multiplies it by CKD5_RBF_SCALE itself.
        "ckd5": _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD5_RBF_SCALE,  # 420 mL/min = 25.2 L/h
    }

    if VERBOSE:
        print(
            f"[constants] Flow maps rebuilt - "
            f"GFR_scale={CKD_SCALE_FACTORS['ckd5']:.3f}, "
            f"CDC_urine_scale={CKD_URINE_SCALE:.2f}, "
            f"RBF_scale={CKD5_RBF_SCALE:.2f}"
        )


# build the maps when the module is imported
_TUBULAR_FLOWS_ML_MIN:     dict = {}
RENAL_BLOOD_FLOW_ML_MIN:   dict = {}
_rebuild_mode_maps()


def set_ckd_scale(scale: float) -> None:
    """
    Change the CKD-5 GFR scale (default 0.10) and rebuild the flow maps.
    For sensitivity analysis. Does not change CKD5_RBF_SCALE or the
    collecting duct flows.
    """
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"CKD scale must be in (0, 1], got {scale}")
    CKD_SCALE_FACTORS["ckd5"] = float(scale)
    _rebuild_mode_maps()
    if VERBOSE:
        print(f"[constants] CKD-5 GFR scale set to {scale:.3f}")


def get_tubular_flows_ml_min(mode: str, scale: float | None = None) -> dict:
    """
    Return a copy of per-segment tubular flows (mL/min) for the given mode.

    Parameters
    ----------
    mode  : "healthy" or "ckd5"
    scale : optional extra multiplier applied to ALL segments.
            For sensitivity analysis only.
    """
    if mode not in _TUBULAR_FLOWS_ML_MIN:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'healthy' or 'ckd5'.")
    flows = _TUBULAR_FLOWS_ML_MIN[mode].copy()
    if scale is not None:
        flows = {k: v * float(scale) for k, v in flows.items()}
    if VERBOSE:
        print("_TUBULAR_FLOWS_ML_MIN[mode]", flows)
    return flows


def get_renal_blood_flow_ml_min(mode: str, scale: float | None = None) -> float:
    """
    Return renal blood flow (mL/min) for the given mode.

    physiology.py calls this with mode="healthy" and applies CKD5_RBF_SCALE
    itself.
    """
    if mode not in RENAL_BLOOD_FLOW_ML_MIN:
        raise ValueError(f"Unknown mode '{mode}'.")
    rbf = RENAL_BLOOD_FLOW_ML_MIN[mode]
    if scale is not None:
        rbf = rbf * float(scale)
    if VERBOSE:
        print("RENAL_BLOOD_FLOW_ML_MIN[mode]", rbf)
    return rbf


# ---------------------------------------------------------------------------
# 6. Segment volumes [L], both kidneys
# ---------------------------------------------------------------------------
# Default volume of the vascular, cell and lumen compartment of each
# segment (physiology.py can give separate values). Same in healthy and
# CKD-5.

V_SEG = {
    "P1":   0.0305 * _twoK_to_oneK,
    "P2":   0.0305 * _twoK_to_oneK,
    "P3":   0.0305 * _twoK_to_oneK,
    "LH1":  0.0027 * _twoK_to_oneK,
    "LH2":  0.0027 * _twoK_to_oneK,
    "DT":   0.0194 * _twoK_to_oneK,
    "CDC1": 0.2370 * _twoK_to_oneK,
    "CDC2": 0.2370 * _twoK_to_oneK,
    "CDC3": 0.2370 * _twoK_to_oneK,
    "CDC4": 0.2370 * _twoK_to_oneK,
    "CDC5": 0.2370 * _twoK_to_oneK,
}


# ---------------------------------------------------------------------------
# 7. pH values (used for the un-ionised fractions in passive diffusion)
# ---------------------------------------------------------------------------
# Lumen pH falls from 7.2 (P1) to 6.5 (collecting duct) because protons are
# secreted. Plasma pH 7.40, cell pH 7.20. The CKD-5 values are the same as
# healthy (metabolic acidosis is not included).

_PH_MAPS = {
    "healthy": {
        "lumen": {
            "P1": 7.2, "P2": 7.1, "P3": 7.0,
            "LH1": 7.0, "LH2": 7.0, "DT": 6.9,
            "CDC1": 6.8, "CDC2": 6.7, "CDC3": 6.6, "CDC4": 6.5, "CDC5": 6.5,
        },
        "plasma": 7.40,
        "cell": {seg: 7.20 for seg in SEGMENTS},
    },
    "ckd5": {
        "lumen": {
            "P1": 7.2, "P2": 7.1, "P3": 7.0,
            "LH1": 7.0, "LH2": 7.0, "DT": 6.9,
            "CDC1": 6.8, "CDC2": 6.7, "CDC3": 6.6, "CDC4": 6.5, "CDC5": 6.5,
        },
        "plasma": 7.40,
        "cell": {seg: 7.20 for seg in SEGMENTS},
    },
}

LUMEN_pH  = _PH_MAPS[MODE]["lumen"]
PLASMA_pH = _PH_MAPS[MODE]["plasma"]
CELL_pH   = _PH_MAPS[MODE]["cell"]


def update_ph_maps(mode: str = "healthy") -> None:
    """Set LUMEN_pH, PLASMA_pH and CELL_pH for the given mode."""
    global LUMEN_pH, PLASMA_pH, CELL_pH
    if mode not in _PH_MAPS:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'healthy' or 'ckd5'.")
    LUMEN_pH  = _PH_MAPS[mode]["lumen"]
    PLASMA_pH = _PH_MAPS[mode]["plasma"]
    CELL_pH   = _PH_MAPS[mode]["cell"]


# ---------------------------------------------------------------------------
# 8. Molecular weights [g/mol]  (mg/L = uM x MW / 1000)
# ---------------------------------------------------------------------------

MW_FUR = 330.74   # furosemide  (C12H11ClN2O5S)
MW_IS  = 213.17   # indoxyl sulfate  (C8H7NO4S)
MW_HA  = 179.17   # hippuric acid  (C9H9NO3)
MW_pCS = 188.19   # p-cresyl sulfate  (C7H8O4S)


# ---------------------------------------------------------------------------
# 9. Proximal tubule geometry
# ---------------------------------------------------------------------------
# Share of the transporter capacity in P1, P2, P3 (40/40/20 %, after the
# axial OAT1 distribution reported by Breljak et al. 2016). The ODE file
# defines the same weights (W_BASO, W_APIC); these copies are used by
# some analysis scripts.

RAW_BASO_PROX = {"P1": 0.40, "P2": 0.40, "P3": 0.20}
RAW_APIC_PROX = {"P1": 0.40, "P2": 0.40, "P3": 0.20}

_sum_baso = sum(RAW_BASO_PROX.values())
_sum_apic = sum(RAW_APIC_PROX.values())

W_BASO_PROX = {k: v / _sum_baso for k, v in RAW_BASO_PROX.items()}
W_APIC_PROX = {k: v / _sum_apic for k, v in RAW_APIC_PROX.items()}

# Proximal tubule: basolateral membrane area = apical area / 30
# (Huang & Isoherranen 2018). Used for the passive diffusion clearances.
PROX_BASO_OVER_APICAL  = 1.0 / 30.0

# Brush border: the apical area is multiplied by this factor
MICROVILLI_FACTOR_APICAL = 10.0


# ---------------------------------------------------------------------------
# 10. Numerical constants and unit conversions
# ---------------------------------------------------------------------------
EPS  = 1e-12   # small number that avoids division by zero
RTOL = 1e-7    # relative tolerance for scipy solve_ivp
ATOL = 1e-9    # absolute tolerance for scipy solve_ivp

ML_MIN_TO_L_H  = 60.0 / 1000.0    # 1 mL/min = 0.06 L/h
DM2_TO_CM2     = 100.0             # 1 dm^2 = 100 cm^2
CM_S_TO_L_H    = 3600.0 / 1000.0  # P [cm/s] x A [cm^2] = cm^3/s; x 3.6 -> L/h


# ---------------------------------------------------------------------------
# 11. Check of the flow maps
# ---------------------------------------------------------------------------

def validate_flows(
    flows_ml_min: dict,
    gfr_ml_min: float,
    urine_flow_ml_min: float,
) -> bool:
    """
    Check the tubular flows:
      1. P1 flow = GFR (within 10 %)
      2. CDC5 flow = urine flow (within 10 %)
      3. flow never increases from one segment to the next
    Returns True if all checks pass; prints a warning for each failure.
    """
    p1    = flows_ml_min["P1"]
    cdc5  = flows_ml_min["CDC5"]
    ok    = True

    # Check 1: P1 ~ GFR
    ratio = p1 / max(gfr_ml_min, EPS)
    if abs(ratio - 1.0) > 0.1:
        print(
            f"[WARN] validate_flows: P1={p1:.3f} mL/min != GFR={gfr_ml_min:.3f} "
            f"(ratio={ratio:.3f})"
        )
        ok = False

    # Check 2: CDC5 ~ urine flow
    ratio2 = cdc5 / max(urine_flow_ml_min, EPS)
    if abs(ratio2 - 1.0) > 0.1:
        print(
            f"[WARN] validate_flows: CDC5={cdc5:.3f} mL/min != "
            f"urine={urine_flow_ml_min:.3f} (ratio={ratio2:.3f})"
        )
        ok = False

    # Check 3: monotonic decrease
    for a, b in zip(SEGMENTS[:-1], SEGMENTS[1:]):
        if flows_ml_min[a] < flows_ml_min[b]:
            print(
                f"[WARN] validate_flows: flow increases from "
                f"{a}({flows_ml_min[a]:.3f}) to {b}({flows_ml_min[b]:.3f})"
            )
            ok = False

    if ok and VERBOSE:
        print("Flow validation passed!")
    return ok


# set the pH maps of the default mode
update_ph_maps(MODE)


# ---------------------------------------------------------------------------
# Quick test: python constants.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== constants.py test ===\n")

    # Test 1: Healthy flows are unchanged
    h = get_tubular_flows_ml_min("healthy")
    assert h["P1"] == 120, f"FAIL: P1={h['P1']}"
    assert h["CDC5"] == 1, f"FAIL: CDC5={h['CDC5']}"
    print(f"  OK Healthy: P1={h['P1']} mL/min  CDC5={h['CDC5']} mL/min")

    # Test 2: CKD5 GFR-scaled segments
    c = get_tubular_flows_ml_min("ckd5")
    assert abs(c["P1"] - 12.0) < 0.01, f"FAIL P1={c['P1']}"
    print(f"  OK CKD5 GFR: P1={c['P1']} mL/min (= 120x0.10)")

    # Test 3: CKD5 CDC5 uses urine scale not GFR scale
    assert abs(c["CDC5"] - 0.5) < 0.01, f"FAIL CDC5={c['CDC5']}"
    print(f"  OK CKD5 urine: CDC5={c['CDC5']} mL/min (= 0.5 -> {c['CDC5']*ML_MIN_TO_L_H:.3f} L/h)")

    # Test 4: CKD5 monotonicity
    segs = ["P1","P2","P3","LH1","LH2","DT","CDC1","CDC2","CDC3","CDC4","CDC5"]
    for a, b in zip(segs[:-1], segs[1:]):
        assert c[a] >= c[b], f"FAIL monotonicity: {a}({c[a]}) < {b}({c[b]})"
    print(f"  OK CKD5 monotonicity: all flows non-increasing")

    # Test 5: DT->CDC1 boundary (no flow increase)
    assert c["CDC1"] <= c["DT"], f"FAIL: CDC1({c['CDC1']}) > DT({c['DT']})"
    print(f"  OK DT->CDC1 boundary: DT={c['DT']} >= CDC1={c['CDC1']}")

    # Test 6: Q_urine values
    Q_urine_h = h["CDC5"] * ML_MIN_TO_L_H
    Q_urine_c = c["CDC5"] * ML_MIN_TO_L_H
    assert abs(Q_urine_h - 0.06) < 1e-6, f"FAIL Q_urine_h={Q_urine_h}"
    assert abs(Q_urine_c - 0.030) < 1e-6, f"FAIL Q_urine_c={Q_urine_c}"
    print(f"  OK Q_urine healthy={Q_urine_h:.3f} L/h, CKD5={Q_urine_c:.3f} L/h")

    # Test 7: RBF values
    rbf_h = get_renal_blood_flow_ml_min("healthy")
    assert abs(rbf_h - 1200.0) < 1, f"FAIL rbf_h={rbf_h}"
    Q_kidney_h = rbf_h * ML_MIN_TO_L_H
    Q_kidney_c = _RENAL_BLOOD_FLOW_BASE_ML_MIN * CKD5_RBF_SCALE * ML_MIN_TO_L_H
    print(f"  OK Q_kidney healthy={Q_kidney_h:.1f} L/h, CKD5={Q_kidney_c:.1f} L/h")

    # Test 8: MW values
    assert abs(MW_IS - 213.17) < 0.01
    assert abs(MW_FUR - 330.74) < 0.01
    print(f"  OK MW_FUR={MW_FUR}, MW_IS={MW_IS}, MW_HA={MW_HA}, MW_pCS={MW_pCS}")

    # Test 9: CKD-5 renal blood flow scale
    assert CKD5_RBF_SCALE == 0.35
    print(f"  OK CKD5_RBF_SCALE={CKD5_RBF_SCALE}")

    # Test 10: validate_flows passes for both modes
    assert validate_flows(h, h["P1"], h["CDC5"])
    VERBOSE = False
    assert validate_flows(c, c["P1"], c["CDC5"])
    VERBOSE = True
    print(f"  OK validate_flows passes for healthy and CKD5")

    print("\n=== All 10 tests passed. ===\n")