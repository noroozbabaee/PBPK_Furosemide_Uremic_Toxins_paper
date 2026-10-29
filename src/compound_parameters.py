"""
compound_parameters.py
----------------------
Parameter values for furosemide and the three uremic toxins

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
This is the ONLY place where compound parameters are defined. Each compound
is a Python dictionary:
    DRUG_BASE  furosemide
    TOX1_BASE  indoxyl sulfate (IS)
    TOX2_BASE  hippuric acid (HA)
    TOX3_BASE  p-cresyl sulfate (pCS)
The dictionaries hold the HEALTHY values. CKD stage 5 values are made by
get_compound_parameters("ckd5"), which copies the healthy dictionaries and
applies:
    1. lower transporter capacity (Vmax x scale factor, see _CKD_SCALES)
    2. different plasma protein binding (fu_p, fu_cell, see _CKD_FU_P)
Everything else (Km, Ki, R0, pKa, k_elim, ...) is the same in both states.

Units
-----
concentration mg/L, Vmax mg/h, clearance L/h, volume L, rate constants 1/h.
Km values from the literature are in uM; they are converted with
    Km [mg/L] = Km [uM] x MW [g/mol] / 1000

Inhibition constant Ki (furosemide -> OAT1 uptake of the toxins)
----------------------------------------------------------------
Ki = 0.35 mg/L, the same value for IS, HA and pCS.
How the value was chosen:
  - For each toxin separately, Ki was varied in a sensitivity analysis
    under healthy conditions.
  - Ki = 0.35 mg/L gives an average increase in toxin plasma
    concentration of about 5 %. The value was selected by visual
    inspection; no formal fitting was done.
  - Values above about 1 mg/L gave little or no interaction.
  - Ki was NOT fitted to clinical data. The clinical study of
    Costes-Albrespic et al. (Kidney Int Rep 2025;10:2165) was only used
    afterwards as a qualitative check.
If you change Ki, change it for all three toxins (Ki_furo_to_*_mgL).

Main literature sources
-----------------------
Protein binding of the toxins (fu_p)
    Deltombe O et al. Toxins 2015;7:3933-3946.
Protein binding of furosemide
    Rane A et al. Clin Pharmacol Ther 1978;24:199-207.
OAT1 Km of the toxins (IS 18 uM, HA 24 uM, pCS 120 uM)
    Chang SY et al. Pharm Res 2023;40:2597-2606.
OAT1 Km of furosemide (38.9 uM)
    Hasannejad H et al. J Pharmacol Exp Ther 2004;308:1021-1029.
Proximal tubule model structure (segments, transporter weighting)
    Huang W, Isoherranen N. CPT Pharmacometrics Syst Pharmacol 2018;7:593-602.
    Huang W, Isoherranen N. CPT Pharmacometrics Syst Pharmacol 2020;9:571-583.

Notes
-----
Values without a reference in the comments are model parameters chosen
for this study (for example apical Vmax values and the CKD scale factors).
They are described in the Methods and Supplementary Material of the paper.
When the module is imported, _self_check() checks the most important
values and prints one line.
"""

from copy import deepcopy
from constants import MW_FUR, MW_IS, MW_HA, MW_pCS


def Km_uM_to_mg_per_L(km_uM: float, mw_g_per_mol: float) -> float:
    """Convert Km from micromolar to mg/L using molecular weight."""
    return float(km_uM * mw_g_per_mol * 1e-3)


# Small helpers used when the CKD-5 values are built.
# _safe_float returns `default` for NaN, +/-inf or values that are not numbers.
def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _scale_value(v, s, clamp_nonneg=True):
    out = _safe_float(v, 0.0) * float(s)
    if clamp_nonneg and out < 0.0:
        out = 0.0
    return out


# ---------------------------------------------------------------------------
# FUROSEMIDE (healthy values)
# ---------------------------------------------------------------------------
def _create_drug_base():
    return {
        "name":  "furosemide",
        "type":  "acid",
        "MW":    MW_FUR,            # 330.74 g/mol
        "pKa":   3.80,              # carboxylic acid; mostly ionised at pH 7.4
        "P_app": 5.0e-7,            # apparent membrane permeability [cm/s],
                                    # used for passive diffusion

        # -- Protein binding -------------------------------------------------
        "BP":      0.82,            # blood-to-plasma concentration ratio
        "fu_p":    0.02,            # unbound fraction in plasma (about 98 %
                                    # bound to albumin; Rane 1978)
        "fu_cell": 0.02,            # unbound fraction inside the cell

        # -- Distribution (two-compartment) ----------------------------------
        # Central volume = phys["V_ven0"] = 5 L (blood), see physiology.py.
        # Steady-state volume of distribution:
        #   Vss = Vcen * (1 + k12/k21) = 5.0 * (1 + 0.033/0.50) = 5.33 L
        # (Vperi does not enter Vss, see TissueDistribution in
        #  ode_system_twocompartment.py)
        "Vd_L":    5.33,            # Vss [L]; printed only, not used in the ODEs
        "Vperi_L": 40.0,            # peripheral compartment volume [L]
        "k12_hinv": 0.033,          # central -> peripheral rate constant [1/h]
        "k21_hinv": 0.50,           # peripheral -> central rate constant [1/h]
        "use_tissue": True,         # False = one-compartment furosemide

        # -- Non-renal elimination -------------------------------------------
        # First-order elimination from the central compartment that
        # lumps all non-renal routes. Same value in healthy and CKD-5.
        # Non-renal clearance = k_elim * Vcen = 0.12 * 5 = 0.6 L/h.
        "k_elim_hinv": 0.12,

        # -- Dosing (default: 120 mg IV over 2 h) ----------------------------
        # scenario_manager.py overwrites these values for each scenario.
        "bolus_mg":  0.0,
        "IR":        60.0,          # infusion rate [mg/h]
        "t_inf_h":   2.0,           # infusion duration [h]
        "t_start_h": 0.05,          # infusion start [h]

        # -- Basolateral uptake (blood -> cell) ------------------------------
        # Km_OAT1 = 38.9 uM x 330.74/1000 = 12.87 mg/L (Hasannejad 2004)
        "Km_baso_uptake_uM":  38.90,
        "Km_baso_uptake":     12.87,
        "Vmax_baso_uptake":   4500.0,   # total basolateral capacity [mg/h]
        # With use_oat3 = False the ODEs use Vmax_OAT1 only.
        # Vmax_OAT1 = 0.6 x Vmax_baso_uptake = 2700 mg/h.
        "Vmax_OAT1":          2700.0,

        # -- OAT3 (switched off in the paper parameter set) ------------------
        "Vmax_baso_OAT3": 0.0,
        "Km_baso_OAT3": 2.42,       # not used while use_oat3 is False
        "use_oat3": False,          # False = OAT1-only uptake of furosemide

        # -- Apical secretion (cell -> lumen) --------------------------------
        "Km_api_secr":   20.0,      # [mg/L]
        "Vmax_api_secr": 5000.0,    # [mg/h]
        "CL_sec_api":    34.0,      # linear secretion clearance [L/h]; only
                                    # used if use_apical_mm is False

        # -- Active reabsorption (not included for furosemide) ---------------
        "CL_reabs_api":  0.0,
        "CL_reabs_baso": 0.0,

        # -- Intracellular metabolism ----------------------------------------
        # Small first-order loss inside the proximal tubule cell [L/h].
        "CL_int_cell": 1.3e-4,

        "enabled": True,
    }


# ---------------------------------------------------------------------------
# INDOXYL SULFATE, IS (healthy values)
# ---------------------------------------------------------------------------
def _create_tox1_base():
    return {
        "name": "indoxyl_sulfate",
        "type": "acid",
        "MW":   MW_IS,              # 213.17 g/mol
        "pKa":  -1.8,               # sulfate conjugate: fully ionised
        "P_app": 5e-7,              # apparent permeability [cm/s]

        # -- Protein binding (Deltombe 2015) ---------------------------------
        # healthy: fu_p = 0.16 ; CKD-5: fu_p = 0.10 (see _CKD_FU_P below)
        "BP":      0.70,            # blood-to-plasma ratio
        "fu_p":    0.16,            # unbound fraction in plasma
        "fu_cell": 0.20,            # unbound fraction inside the cell

        # Not used in the ODEs: all toxins use the central volume V_ven0.
        "Vd_L": 14.0,

        # -- Production --------------------------------------------------------
        # R0 is a constant endogenous production rate [mg/h], the SAME in
        # healthy and CKD-5. Higher plasma levels in CKD are therefore caused
        # only by lower renal clearance.
        # At steady state: C_ss = R0 / CL_renal.
        "R0":  5.500,
        "C0":  0.5,                 # initial central concentration [mg/L]

        # -- OAT1 uptake (blood -> cell) -------------------------------------
        # Km = 18 uM x 213.17/1000 = 3.84 mg/L (Chang 2023)
        "Km_baso_uptake_uM":  18.0,
        "Km_baso_uptake":      3.84,
        "Vmax_baso_uptake":  165.0,  # [mg/h]

        # -- Apical secretion (cell -> lumen) --------------------------------
        "Km_api_secr":   10.0,
        "Vmax_api_secr": 180.0,
        "CL_sec_api":    0.0,

        # -- Not included for the toxins -------------------------------------
        "CL_reabs_api":  0.0,
        "CL_reabs_baso": 0.0,
        "CL_int_cell":   0.0,

        # -- Competitive inhibition of OAT1 uptake by furosemide -------------
        # Ki = 0.35 mg/L (same for IS, HA, pCS); see the header of this file.
        "Ki_furo_to_IS_mgL": 0.35,
        # Inhibition between toxins is not modelled (very large Ki = no effect).
        "Ki_IS_to_IS_mgL":   1e9,
        "Ki_HA_to_IS_mgL":   1e9,
    }


# ---------------------------------------------------------------------------
# HIPPURIC ACID, HA (healthy values)
# ---------------------------------------------------------------------------
def _create_tox2_base():
    return {
        "name": "hippuric_acid",
        "type": "acid",
        "MW":   MW_HA,              # 179.17 g/mol
        "pKa":  3.59,               # carboxylic acid
        "P_app": 5e-7,

        # -- Protein binding (Deltombe 2015) ---------------------------------
        # healthy: fu_p = 0.66 ; CKD-5: fu_p = 0.52 (see _CKD_FU_P below)
        "BP":      0.84,
        "fu_p":    0.66,
        "fu_cell": 0.70,

        # Not used in the ODEs: all toxins use the central volume V_ven0.
        "Vd_L": 18.0,

        # -- Production (constant, same in healthy and CKD-5) ----------------
        "R0":  70.0,                # [mg/h]
        "C0":  3.0,                 # initial central concentration [mg/L]

        # -- OAT1 uptake (blood -> cell) -------------------------------------
        # Km = 24 uM x 179.17/1000 = 4.302 mg/L (Chang 2023)
        "Km_baso_uptake_uM":  24.0,
        "Km_baso_uptake":    4.302,
        "Vmax_baso_uptake":  350.0,

        # -- Apical secretion (cell -> lumen) --------------------------------
        "Km_api_secr":    10.0,
        "Vmax_api_secr": 1100.0,
        "CL_sec_api":    0.0,
        "CL_reabs_api":  0.0,
        "CL_int_cell":   0.0,

        # -- Competitive inhibition by furosemide (Ki = 0.35 mg/L) -----------
        "Ki_furo_to_HA_mgL": 0.35,
        "Ki_IS_to_HA_mgL":   1e9,
        "Ki_HA_to_HA_mgL":   1e9,
    }


# ---------------------------------------------------------------------------
# p-CRESYL SULFATE, pCS (healthy values)
# ---------------------------------------------------------------------------
def _create_tox3_base():
    return {
        "name": "p_cresyl_sulfate",
        "type": "acid",
        "MW":   MW_pCS,             # 188.19 g/mol
        "pKa":  -2,                 # sulfate conjugate: fully ionised
        "P_app": 5e-7,

        # -- Protein binding (Deltombe 2015) ---------------------------------
        # healthy: fu_p = 0.06 ; CKD-5: fu_p = 0.05 (see _CKD_FU_P below)
        "BP":      0.55,
        "fu_p":    0.06,
        "fu_cell": 0.20,

        # Not used in the ODEs: all toxins use the central volume V_ven0.
        "Vd_L": 10.0,

        # -- Production (constant, same in healthy and CKD-5) ----------------
        "R0":  2.1,                 # [mg/h]
        "C0":  0.7,                 # initial central concentration [mg/L]

        # -- OAT1 uptake (blood -> cell) -------------------------------------
        # Km = 120 uM x 188.19/1000 = 22.58 mg/L (Chang 2023)
        "Km_baso_uptake_uM": 120.0,
        "Km_baso_uptake":    22.58,
        "Vmax_baso_uptake":  460.0,

        # -- Apical secretion (cell -> lumen) --------------------------------
        "Km_api_secr":   35.0,
        "Vmax_api_secr": 400.0,
        "CL_sec_api":    0.0,
        "CL_reabs_api":  0.0,
        "CL_reabs_baso": 0.0,
        "CL_int_cell":   0.0,

        # -- Competitive inhibition by furosemide (Ki = 0.35 mg/L) -----------
        "Ki_furo_to_pCS_mgL": 0.35,
        "Ki_IS_to_pCS_mgL":   1e9,
        "Ki_HA_to_pCS_mgL":   1e9,
    }


# Healthy dictionaries, created once when the module is imported.
DRUG_BASE = _create_drug_base()
TOX1_BASE = _create_tox1_base()
TOX2_BASE = _create_tox2_base()
TOX3_BASE = _create_tox3_base()


# ---------------------------------------------------------------------------
# CKD stage 5: transporter capacity
# ---------------------------------------------------------------------------
# In CKD-5 every Vmax is multiplied by a scale factor:
#     Vmax_CKD5 = scale x Vmax_healthy
# basolateral (uptake) scale = 0.05, apical (secretion) scale = 0.15,
# for furosemide and for all three toxins. These are model assumptions
# that represent the loss of functional proximal tubule mass and transporter
# activity in CKD-5. Use set_ckd_vmax_scales() to change them, for example
# in a sensitivity analysis.
# Lookup order: "per_compound" entry first, then the "tox"/"drug" default.

_CKD_SCALES = {
    "tox": {
        "baso":   0.050,     # default for toxins not listed in per_compound
        "apical": 0.150,
    },
    "drug": {
        "baso":   0.05,
        "apical": 0.15,
    },
    "per_compound": {
        "IS":  {"baso": 0.05, "apical": 0.150},
        "HA":  {"baso": 0.05, "apical": 0.150},
        "pCS": {"baso": 0.05, "apical": 0.150},
    },
}

_VMAX_KEYS = {
    "baso":   ["Vmax_baso_uptake", "Vmax_baso_uptake_pmol_min_per_million",
               "Vmax_baso_OAT3", "Vmax_OAT1"],
    "apical": ["Vmax_api_secr",    "Vmax_api_secr_pmol_min_per_million"],
}


# ---------------------------------------------------------------------------
# CKD stage 5: protein binding
# ---------------------------------------------------------------------------
# Toxins (Deltombe 2015): the unbound fraction is LOWER in CKD-5 than in
# healthy subjects (IS 0.16 -> 0.10, HA 0.66 -> 0.52, pCS 0.06 -> 0.05).
# Furosemide: the unbound fraction is HIGHER in uremia (Rane 1978);
# 0.02 -> 0.05 in the model.
# fu_cell is changed in proportion to fu_p (fu_cell/fu_p kept constant).
# R0 is repeated here with the healthy value, so production does not change.
# k_elim of furosemide is not changed in CKD-5 (stays 0.12 1/h).

_CKD_FU_P = {
    "drug": {
        "fu_p":       0.05,
        "fu_cell":    0.05,
    },
    "IS":  {"fu_p": 0.10, "fu_cell": 0.1250, "R0": 5.5},
    "HA":  {"fu_p": 0.52, "fu_cell": 0.5515, "R0": 70},
    "pCS": {"fu_p": 0.05, "fu_cell": 0.1667, "R0":  2.10},
}


def _create_ckd_adjustments_dynamic(scales: dict) -> dict:
    """
    Build the CKD-5 changes for each compound.

    Returns a dictionary {"drug": {...}, "IS": {...}, "HA": {...},
    "pCS": {...}} that holds ONLY the parameters that differ from healthy
    (scaled Vmax values and the protein binding values of _CKD_FU_P).
    get_compound_parameters("ckd5") applies them with dict.update().
    """
    baseline = {
        "drug": deepcopy(DRUG_BASE),
        "IS":   deepcopy(TOX1_BASE),
        "HA":   deepcopy(TOX2_BASE),
        "pCS":  deepcopy(TOX3_BASE),
    }
    per_compound_scale = scales.get("per_compound", {}) or {}
    out = {}

    for cmpd in ("drug", "IS", "HA", "pCS"):
        src           = baseline[cmpd]
        default_block = scales.get("drug" if cmpd == "drug" else "tox", {})
        sc_baso = float(per_compound_scale.get(cmpd, {}).get(
                        "baso", default_block.get("baso", 1.0)))
        sc_apic = float(per_compound_scale.get(cmpd, {}).get(
                        "apical", default_block.get("apical", 1.0)))
        adj = {}

        for k in _VMAX_KEYS["baso"]:
            if k in src:
                adj[k] = _scale_value(src[k], sc_baso)
        for k in _VMAX_KEYS["apical"]:
            if k in src:
                adj[k] = _scale_value(src[k], sc_apic)
        if "CL_sec_api" in src:
            adj["CL_sec_api"] = _safe_float(src["CL_sec_api"], 0.0)

        for k, v in _CKD_FU_P.get(cmpd, {}).items():
            adj[k] = float(v)

        out[cmpd] = adj

    return out


def set_ckd_vmax_scales(
    baso_scale=None, apical_scale=None, per_compound_scale=None,
    drug_baso_scale=None, drug_apical_scale=None,
):
    """Override CKD-5 Vmax scale factors and rebuild CKD_ADJUSTMENTS."""
    global _CKD_SCALES, CKD_ADJUSTMENTS
    if baso_scale        is not None: _CKD_SCALES["tox"]["baso"]   = float(baso_scale)
    if apical_scale      is not None: _CKD_SCALES["tox"]["apical"] = float(apical_scale)
    if drug_baso_scale   is not None: _CKD_SCALES["drug"]["baso"]  = float(drug_baso_scale)
    if drug_apical_scale is not None: _CKD_SCALES["drug"]["apical"]= float(drug_apical_scale)
    if per_compound_scale is not None: _CKD_SCALES["per_compound"] = per_compound_scale
    CKD_ADJUSTMENTS = _create_ckd_adjustments_dynamic(_CKD_SCALES)


CKD_ADJUSTMENTS = _create_ckd_adjustments_dynamic(_CKD_SCALES)


# ---------------------------------------------------------------------------
# Main function used by the rest of the code
# ---------------------------------------------------------------------------
def get_compound_parameters(mode: str = "healthy"):
    """
    Return (drug, tox1, tox2, tox3) parameter dicts for the given mode.

    Parameters
    ----------
    mode : "healthy" | "ckd5"

    Returns
    -------
    tuple of 4 dicts: (drug, IS, HA, pCS)

    The dictionaries are new copies, so changing them does not change
    DRUG_BASE, TOX1_BASE, ... Example:

        drug, IS, HA, pCS = get_compound_parameters("ckd5")
        print(IS["fu_p"])          # 0.10
    """
    bases = [
        deepcopy(DRUG_BASE),
        deepcopy(TOX1_BASE),
        deepcopy(TOX2_BASE),
        deepcopy(TOX3_BASE),
    ]
    keys = ["drug", "IS", "HA", "pCS"]

    if mode.lower() in ("ckd5", "ckd-5", "ckd"):
        for base, key in zip(bases, keys):
            base.update(CKD_ADJUSTMENTS[key])
    elif mode.lower() != "healthy":
        raise ValueError(f"Unknown mode: '{mode}'. Must be 'healthy' or 'ckd5'.")

    # Derive Km_baso_uptake [mg/L] from uM where only the uM key is present
    for base in bases:
        if "Km_baso_uptake" not in base and "Km_baso_uptake_uM" in base and "MW" in base:
            base["Km_baso_uptake"] = Km_uM_to_mg_per_L(
                base["Km_baso_uptake_uM"], base["MW"]
            )

    return tuple(bases)


# ---------------------------------------------------------------------------
# Printed summaries (useful to check what the model is using)
# ---------------------------------------------------------------------------

def print_compound_summary(mode: str = "healthy"):
    """
    Print the main parameters with units for the given mode
    ("healthy" or "ckd5").
    """
    drug, IS, HA, pCS = get_compound_parameters(mode)
    W = 68
    label = mode.upper()

    print()
    print("=" * W)
    print(f"  COMPOUND PARAMETER SUMMARY  -  mode: {label}")
    print("=" * W)

    # -- DRUG ----------------------------------------------------------------
    print(f"\n  {'-'*W}")
    print(f"  FUROSEMIDE (drug)")
    print(f"  {'-'*W}")
    print(f"  {'MW':<28} {drug['MW']:.2f} g/mol")
    print(f"  {'BP (blood/plasma)':<28} {drug['BP']:.3f}")
    print(f"  {'fu_p (unbound fraction)':<28} {drug['fu_p']:.4f}   "
          f"({'%.0f' % ((1-drug['fu_p'])*100)}% bound)")
    print(f"  {'fu_cell':<28} {drug['fu_cell']:.4f}")
    print(f"  {'Vd_ss [L]':<28} {drug.get('Vd_L', 'n/a')}   "
          f"(Vcen={5.0} L x (1 + k12/k21), k12={drug['k12_hinv']}, k21={drug['k21_hinv']})")
    print(f"  {'k_elim [h-1]':<28} {drug['k_elim_hinv']:.4f}  "
          f"(t1/2 = {0.693/drug['k_elim_hinv']:.1f} h)")
    print(f"  {'IR [mg/h]':<28} {drug['IR']:.1f}   "
          f"({drug['t_inf_h']:.0f}h infusion = {drug['IR']*drug['t_inf_h']:.0f} mg total)")
    print(f"  {'Km_OAT1 [mg/L]':<28} {drug['Km_baso_uptake']:.2f}  "
          f"({drug['Km_baso_uptake_uM']:.1f} uM)")
    print(f"  {'Vmax_baso_uptake [mg/h]':<28} {drug['Vmax_baso_uptake']:.1f}")
    print(f"  {'Vmax_OAT1 [mg/h]':<28} {drug.get('Vmax_OAT1', 'n/a')}   "
          f"(used when use_oat3 is False)")
    print(f"  {'Vmax_OAT3 [mg/h]':<28} {drug.get('Vmax_baso_OAT3', 'n/a')}")
    print(f"  {'Km_OAT3 [mg/L]':<28} {drug.get('Km_baso_OAT3', 'n/a')}")

    # -- TOXINS --------------------------------------------------------------
    tox_data = [
        ("INDOXYL SULFATE (IS)",   IS,  "Ki_furo_to_IS_mgL",
         "initial value"),
        ("HIPPURIC ACID (HA)",     HA,  "Ki_furo_to_HA_mgL",
         "initial value"),
        ("p-CRESYL SULFATE (pCS)", pCS, "Ki_furo_to_pCS_mgL",
         "initial value"),
    ]

    for tox_name, tox, ki_key, target_note in tox_data:
        print(f"\n  {'-'*W}")
        print(f"  {tox_name}")
        print(f"  {'-'*W}")
        print(f"  {'MW':<28} {tox['MW']:.2f} g/mol")
        print(f"  {'BP (blood/plasma)':<28} {tox['BP']:.3f}")
        print(f"  {'fu_p (unbound fraction)':<28} {tox['fu_p']:.4f}   "
              f"({'%.0f' % ((1-tox['fu_p'])*100)}% bound)")
        print(f"  {'fu_cell':<28} {tox['fu_cell']:.4f}")
        print(f"  {'Vd [L]':<28} {tox.get('Vd_L', 'n/a')}   "
              f"(not used in the ODEs)")
        print(f"  {'R0 [mg/h]':<28} {tox['R0']:.3f}   "
              f"(constant production rate)")
        print(f"  {'C0 [mg/L]':<28} {tox['C0']:.2f}   "
              f"({target_note})")
        print(f"  {'Km_OAT1 [mg/L]':<28} {tox['Km_baso_uptake']:.4f}  "
              f"({tox['Km_baso_uptake_uM']:.0f} uM)")
        print(f"  {'Vmax_OAT1 [mg/h]':<28} {tox['Vmax_baso_uptake']:.1f}")
        print(f"  {'Ki_furo [mg/L]':<28} {tox[ki_key]:.3f}   "
              f"(from sensitivity analysis, healthy)")

    print()
    print("=" * W)
    print()


def print_parameter_changes():
    """Print a table of all parameters that differ between healthy and CKD5."""
    drug_h, t1_h, t2_h, t3_h = get_compound_parameters("healthy")
    drug_c, t1_c, t2_c, t3_c = get_compound_parameters("ckd5")
    print()
    print("=" * 65)
    print("  PARAMETER CHANGES: CKD5 vs HEALTHY")
    print("=" * 65)
    for name, h, c in [
        ("Drug (furosemide)", drug_h, drug_c),
        ("IS",               t1_h,   t1_c),
        ("HA",               t2_h,   t2_c),
        ("pCS",              t3_h,   t3_c),
    ]:
        print(f"\n  {name}:")
        changed = False
        for k in sorted(h.keys()):
            if k in c and h.get(k) != c.get(k):
                changed = True
                hv, cv = h[k], c[k]
                if isinstance(hv, (int, float)) and hv not in (0, 1e9):
                    pct = (cv - hv) / hv * 100.0
                    print(f"    {k:<30} {hv:>10.4f}  ->  {cv:>10.4f}   ({pct:+.1f}%)")
                else:
                    print(f"    {k:<30} {hv!r:>10}  ->  {cv!r:>10}")
        if not changed:
            print("    (no changes from healthy baseline)")
    print()


def print_healthy_summary():
    """Short form of print_compound_summary("healthy")."""
    print_compound_summary("healthy")


def print_ckd5_summary():
    """Short form of print_compound_summary("ckd5")."""
    print_compound_summary("ckd5")


# ---------------------------------------------------------------------------
# Check at import time
# ---------------------------------------------------------------------------
def _self_check():
    """
    Compare the most important parameter values with the values used in the
    paper. Prints one line if everything matches; otherwise prints every
    value that differs. A difference is not an error if you changed a
    parameter on purpose (for example in a sensitivity analysis).
    """
    d, IS, HA, pCS = get_compound_parameters("healthy")
    dc, ISc, HAc, pCSc = get_compound_parameters("ckd5")

    checks = [
        ("IS  fu_p healthy",   IS["fu_p"],                    0.16),
        ("HA  fu_p healthy",   HA["fu_p"],                    0.66),
        ("pCS fu_p healthy",   pCS["fu_p"],                   0.06),
        ("IS  fu_p CKD5",      ISc["fu_p"],                   0.10),
        ("HA  fu_p CKD5",      HAc["fu_p"],                   0.52),
        ("pCS fu_p CKD5",      pCSc["fu_p"],                  0.05),
        ("drug fu_p CKD5",     dc["fu_p"],                    0.05),
        ("drug k_elim CKD5",   dc["k_elim_hinv"],             0.12),
        ("drug Vmax_baso CKD5",dc["Vmax_baso_uptake"],        225.0),  # 4500*0.05
        ("drug Vmax_api  CKD5",dc["Vmax_api_secr"],           750.0),  # 5000*0.15
        ("IS  Ki_furo",        IS["Ki_furo_to_IS_mgL"],       0.35),
        ("HA  Ki_furo",        HA["Ki_furo_to_HA_mgL"],       0.35),
        ("pCS Ki_furo",        pCS["Ki_furo_to_pCS_mgL"],     0.35),
        ("IS  Km_OAT1",        IS["Km_baso_uptake"],          3.84),
        ("pCS pKa",            pCS["pKa"],                   -2.0),
        ("IS  pKa",            IS["pKa"],                    -1.8),
        ("HA  pKa",            HA["pKa"],                     3.59),
        ("IS  R0",             IS["R0"],                      5.5),
        ("HA  R0",             HA["R0"],                     70.0),
        ("pCS R0",             pCS["R0"],                     2.1),
        ("drug k_elim healthy",d["k_elim_hinv"],              0.12),
    ]

    all_ok = True
    for label, actual, expected in checks:
        # relative tolerance 1 % (absolute 1e-6 for values close to zero)
        ok = abs(float(actual) - float(expected)) <= max(0.01 * abs(float(expected)), 1e-6)
        if not ok:
            all_ok = False
            print(f"  [compound_parameters] {label:<22} = {actual}  (paper value {expected})")

    if all_ok:
        print("[compound_parameters] parameter values match the paper set.")


_self_check()


__all__ = [
    "DRUG_BASE", "TOX1_BASE", "TOX2_BASE", "TOX3_BASE",
    "CKD_ADJUSTMENTS",
    "get_compound_parameters",
    "set_ckd_vmax_scales",
    "print_compound_summary",
    "print_parameter_changes",
    "print_healthy_summary",
    "print_ckd5_summary",
    "Km_uM_to_mg_per_L",
]
