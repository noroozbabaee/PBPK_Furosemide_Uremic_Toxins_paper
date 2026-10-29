"""
helpers.py
----------
Small helper functions (unit conversion, ionisation, surface areas)

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
Functions used by several modules:
  - Km_uM_to_mg_per_L, Ki_uM_to_mgL : uM -> mg/L
  - frac_unionized                  : Henderson-Hasselbalch
  - derive_baso_from_apical         : basolateral membrane areas
  - drug_baso_uptake_flux_unb       : basolateral uptake of furosemide
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional


# ---------------------------------------------------------------------------
# Numeric helpers: convert to float, return a default for NaN/inf/None
# ---------------------------------------------------------------------------

def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _f(x, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Unit conversions (MW in g/mol)
# ---------------------------------------------------------------------------

def Km_uM_to_mg_per_L(km_uM: float, mw_g_per_mol: float) -> float:
    """uM -> mg/L: Km[mg/L] = Km[uM] * MW[g/mol] / 1000."""
    km = _f(km_uM)
    mw = _f(mw_g_per_mol)
    return km * mw * 1e-3


def Ki_uM_to_mgL(ki_uM: float, mw_g_per_mol: float) -> float:
    """uM -> mg/L: Ki[mg/L] = Ki[uM] * MW[g/mol] / 1000."""
    ki = _f(ki_uM)
    mw = _f(mw_g_per_mol)
    return ki * mw * 1e-3


def Vmax_pmolmin_per_million_to_mgph_per_million(vmax_pmol_min_per_million: float,
                                                 mw_g_per_mol: float) -> float:
    """
    pmol/min/1e6 cells -> mg/h/1e6 cells
    Factor: 1 pmol = 1e-12 mol; min->h: *60; g->mg: *1000
    => pmol/min * MW[g/mol] * 60 * 1e-12 * 1000 = pmol/min * MW * 6e-8
    """
    vmax = _f(vmax_pmol_min_per_million)
    mw = _f(mw_g_per_mol)
    return vmax * mw * 6e-8


# ---------------------------------------------------------------------------
# Henderson-Hasselbalch: fraction of the compound that is NOT ionised
# ---------------------------------------------------------------------------
# Only the un-ionised form crosses membranes by passive diffusion.
# Example: furosemide (acid, pKa 3.8) at pH 7.4 -> 1/(1+10^3.6) = 0.00025

def frac_unionized(drug_type: str, pKa: float, pH: float) -> float:
    """
    Unionized fraction for a monoprotic acid/base.

    acid:  fu = 1 / (1 + 10^(pH - pKa))
    base:  fu = 1 / (1 + 10^(pKa - pH))
    other/unknown: returns 1.0
    """
    t = (drug_type or "").strip().lower()
    pka = _f(pKa, default=float("nan"))
    ph = _f(pH, default=float("nan"))

    if not (_isfinite(pka) and _isfinite(ph)):
        return 1.0

    # limit the exponent to avoid overflow for extreme pKa values
    def _pow10(x: float) -> float:
        x = max(min(x, 50.0), -50.0)
        return 10.0 ** x

    if t == "acid":
        return 1.0 / (1.0 + _pow10(ph - pka))
    if t == "base":
        return 1.0 / (1.0 + _pow10(pka - ph))
    return 1.0


# ---------------------------------------------------------------------------
# Membrane surface areas
# ---------------------------------------------------------------------------
# In the proximal tubule the basolateral area is taken as 1/30 of the
# apical area (brush border), as in Huang & Isoherranen 2018.
# For the other segments basolateral area = apical area.

def derive_baso_from_apical(
    sa_apical_dm2: Dict[str, float],
    *,
    proximal_segments: Iterable[str] = ("P1", "P2", "P3"),
    prox_baso_over_apical: float = 1.0 / 30.0
) -> Dict[str, float]:
    """
    Basolateral area map (dm^2) from apical:
      - proximal: baso = apical * prox_baso_over_apical
      - others:   baso = apical
    """
    prox = set(proximal_segments)
    ratio = _f(prox_baso_over_apical, default=1.0 / 30.0)
    out: Dict[str, float] = {}

    for seg, a in (sa_apical_dm2 or {}).items():
        ap = _f(a, default=0.0)
        out[seg] = ap * (ratio if seg in prox else 1.0)

    return out


def get_sa_baso_dm2() -> Dict[str, float]:
    """
    Basolateral areas computed from the current constants.SA_APICAL_DM2
    (so a change of the apical areas at runtime is included).
    """
    import constants as K
    return derive_baso_from_apical(
        K.SA_APICAL_DM2,
        proximal_segments=getattr(K, "PROX", ("P1", "P2", "P3")),
        prox_baso_over_apical=getattr(K, "PROX_BASO_OVER_APICAL", 1.0 / 30.0),
    )


# ---------------------------------------------------------------------------
# Furosemide helpers
# ---------------------------------------------------------------------------

def drug_on() -> bool:
    """True iff compounds.drug exists and is enabled."""
    try:
        import compounds as C
        return bool(getattr(C, "drug", None)) and bool(C.drug.get("enabled", False))
    except Exception:
        return False

def drug_baso_uptake_flux_unb(Cv_unb: float, seg: Optional[str] = None) -> float:
    """
    Basolateral uptake of furosemide, blood -> cell [mg/h].

    Michaelis-Menten with the TOTAL basolateral capacity:
        J = Vmax_baso_uptake * Cu / (Km_baso_uptake + Cu)
    or linear (CL_scr_baso * Cu) if drug["baso_model"] == "linear".
    Used by the ODE only when drug["use_oat3"] is True; the paper
    parameter set uses the OAT1-only path in ode_system_twocompartment.py.

    Cv_unb : unbound furosemide concentration in the vascular segment [mg/L]
    seg    : not used
    """
    try:
        import compounds as C
        import constants as K
    except Exception:
        return 0.0

    drug = getattr(C, "drug", None)
    if not isinstance(drug, dict) or not drug:
        return 0.0

    Cv = _f(Cv_unb, default=0.0)
    if Cv <= 0.0:
        return 0.0

    eps = float(getattr(K, "EPS", 1e-12))

    if (drug.get("baso_model", "mm") or "").strip().lower() == "linear":
        return _f(drug.get("CL_scr_baso", 0.0)) * Cv

    vmax = _f(drug.get("Vmax_baso_uptake", 0.0))
    km = _f(drug.get("Km_baso_uptake", 0.0))
    return (vmax * Cv) / (km + Cv + eps)
