"""
passive_diffusion.py
--------------------
Passive diffusion across the tubular cell membranes

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
build_passive_params() computes, for one compound, the passive diffusion
clearances and the un-ionised fractions for every nephron segment.
compounds.py calls it once per compound; the result (DRG_PASS, TOX1_PASS,
...) is used in the ODE functions _pd_baso() and _pd_apic().

Approach (Huang & Isoherranen, CPT Pharmacometrics Syst Pharmacol
2018;7:593-602 and 2020;9:571-583)
----------------------------------
Only the un-ionised, unbound form diffuses. The flux over a membrane is
    J = CL_PD x (fraction un-ionised x unbound concentration, side 1
                 - fraction un-ionised x unbound concentration, side 2)
The toxins are sulfate conjugates or a carboxylic acid and are almost
fully ionised at physiological pH, so their passive flux is very small.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from helpers import frac_unionized, derive_baso_from_apical
import constants as K


def build_passive_params(
    prefix: str,
    cpd: Mapping[str, float],
    *,
    mode: str = "healthy",
    sa_apical_dm2: Optional[Mapping[str, float]] = None,
    sa_baso_dm2: Optional[Mapping[str, float]] = None,
    proximal_segments: Sequence[str] = K.PROX,
    prox_baso_over_apical: float = K.PROX_BASO_OVER_APICAL,
    microvilli_factor_apical: float = K.MICROVILLI_FACTOR_APICAL,
) -> Dict[str, float]:
    """
    Passive diffusion parameters of one compound for every segment.

    For each segment and membrane side the clearance of the un-ionised form is
        CL [L/h] = P_app [cm/s] x area [cm^2] x 3600 s/h / 1000 cm^3/L
    The apical area of the proximal segments is multiplied by the
    microvilli factor (brush border). The un-ionised fractions are
        alpha (lumen pH), beta (cell pH), gamma (plasma pH)
    and depend on the mode, because lumen pH differs between healthy and
    CKD-5 (constants.update_ph_maps).

    Required keys in `cpd`
    ----------------------
    - "P_app": intrinsic permeability (cm/s)
    - "type": acid/base/neutral definition used by helpers.frac_unionized
    - "pKa": pKa value (float)

    Returns
    -------
    Dict[str, float] with keys like:
      {prefix}_alpha_{seg}, {prefix}_beta_{seg}, {prefix}_gamma_plasma,
      {prefix}_CL_PD_{seg}_apical_no_ui, {prefix}_CL_PD_{seg}_basolateral_no_ui,
      {prefix}_CL_PD_apical_TOTAL_no_ui, {prefix}_CL_PD_basolateral_TOTAL_no_ui,
      {prefix}_mode
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in {"healthy", "ckd5"}:
        raise ValueError("mode must be 'healthy' or 'ckd5'")

    # set the pH values of the requested mode
    K.update_ph_maps(mode_key)

    if sa_apical_dm2 is None:
        sa_apical_dm2 = K.SA_APICAL_DM2
    if sa_baso_dm2 is None:
        sa_baso_dm2 = derive_baso_from_apical(
            sa_apical_dm2,
            proximal_segments=tuple(proximal_segments),
            prox_baso_over_apical=float(prox_baso_over_apical),
        )

    P_int = float(cpd["P_app"])  # cm/s

    lumen_pH = K.LUMEN_pH
    cell_pH = K.CELL_pH
    plasma_pH = K.PLASMA_pH

    out: Dict[str, float] = {}

    # un-ionised fraction in plasma (same for all segments)
    gamma = frac_unionized(cpd["type"], cpd["pKa"], plasma_pH)
    out[f"{prefix}_gamma_plasma"] = float(gamma)

    ap_tot = 0.0
    bs_tot = 0.0

    for seg in sa_apical_dm2.keys():
        if seg not in lumen_pH or seg not in cell_pH:
            raise KeyError(f"Segment {seg!r} missing from pH maps.")

        # un-ionised fractions in lumen (alpha) and cell (beta)
        alpha = frac_unionized(cpd["type"], cpd["pKa"], lumen_pH[seg])
        beta = frac_unionized(cpd["type"], cpd["pKa"], cell_pH[seg])

        out[f"{prefix}_alpha_{seg}"] = float(alpha)
        out[f"{prefix}_beta_{seg}"] = float(beta)

        # areas in cm^2 and clearances in L/h
        ap_cm2 = float(sa_apical_dm2[seg]) * K.DM2_TO_CM2 * float(microvilli_factor_apical)
        bs_cm2 = float(sa_baso_dm2[seg]) * K.DM2_TO_CM2

        # (cm/s) x (cm^2) = cm^3/s ; x 3600 / 1000 -> L/h (K.CM_S_TO_L_H)
        CL_ap_no = P_int * ap_cm2 * K.CM_S_TO_L_H
        CL_bs_no = P_int * bs_cm2 * K.CM_S_TO_L_H

        out[f"{prefix}_CL_PD_{seg}_apical_no_ui"] = float(CL_ap_no)
        out[f"{prefix}_CL_PD_{seg}_basolateral_no_ui"] = float(CL_bs_no)

        ap_tot += float(CL_ap_no)
        bs_tot += float(CL_bs_no)

    out[f"{prefix}_CL_PD_apical_TOTAL_no_ui"] = float(ap_tot)
    out[f"{prefix}_CL_PD_basolateral_TOTAL_no_ui"] = float(bs_tot)
    out[f"{prefix}_mode"] = mode_key

    return out
