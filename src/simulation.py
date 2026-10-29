"""
simulation.py
-------------
Run a simulation of the kidney PBPK model

Author      : Leyla Noroozbabaee
Affiliation : Maastricht Centre for Systems Biology (MaCSBio),
              Faculty of Science and Engineering, Maastricht University,
              Maastricht, The Netherlands
Date        : October 2026

Purpose
-------
run_sim() is the function you call to simulate the model. It
  1. checks the compound dictionaries (compounds.validate_compounds),
  2. prints a parameter check table (parameter_audit.audit_compounds),
  3. builds the physiology and the tubular flows (physiology.py),
  4. builds the initial state vector (state_vector.py),
  5. solves the ODEs with scipy.integrate.solve_ivp (Radau method),
  6. returns the solution object (sol.t, sol.y).

Typical use (see scenario_manager.py for the full workflow)
-----------------------------------------------------------
    from simulation import run_sim
    from state_vector import IDX

    sol = run_sim(48.0, mode="healthy", dt=0.05)
    C_furo = sol.y[IDX["C_ven"]]        # furosemide, central [mg/L]
    C_IS   = sol.y[IDX["C_tox1_cen"]]   # indoxyl sulfate, central [mg/L]

Units: concentration mg/L, flow L/h, time h, excreted amount Ae mg.

Solver
------
Radau is an implicit method for stiff ODE systems. The kidney model is
stiff because the segment volumes are very small compared with the central
volume, so some states change much faster than others. Tolerances are
constants.RTOL and constants.ATOL.
When furosemide is infused, the run is split in two parts (during and after
the infusion) because the infusion rate jumps to zero at the end of the
infusion; solving the two smooth parts separately is more accurate.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

# 'import constants as K' (instead of 'from constants import ...') makes sure
# that a flag changed at runtime, e.g. K.VERBOSE = False, is seen here.
import constants as K
import compounds as C
from physiology import make_phys, flows_L_per_h
from state_vector import initial_conditions, IDX


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_sim(
    t_end_h: float,
    custom_initial_conds: Optional[Dict[str, float]] = None,
    mode: str = "healthy",
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    **kwargs,
):
    """
    Solve the kidney PBPK model from t = 0 to t_end_h.

    Parameters
    ----------
    t_end_h : float
        Simulation end time [h].

    custom_initial_conds : dict or None
        Override specific initial conditions. Keys are state names (from IDX),
        values are concentrations [mg/L] or amounts [mg] for Ae states.
        Any state not in this dict uses the default from initial_conditions().
        Typical use: start from the end state of a longer run without drug
        (toxins at steady state). Cumulative amounts Ae_* are reset to 0.

    mode : str
        Physiological mode. "healthy" or "ckd5".
        Sets GFR, renal blood flow and urine flow (physiology.py).
        The compound parameters for the mode must be loaded before
        (compounds.initialize_compounds(mode)).

    dt : float or None
        Time step of the OUTPUT grid [h] (the solver chooses its own
        internal steps). Used to compute n_steps if n_steps is not given.

    n_steps : int or None
        Number of time points in the output grid. If both dt and n_steps
        are None, defaults to 10000 steps.

    rtol, atol : float
        Relative and absolute tolerances for the Radau solver.
        Note: the solver uses constants.RTOL and constants.ATOL; these two
        arguments are kept for compatibility but are not passed on.

    **kwargs : dict
        audit_strict (bool, default False): passed to audit_compounds().

    Returns
    -------
    scipy.integrate.OdeResult
        sol.t  : time vector [h], shape (n_steps+1,)
        sol.y  : state matrix [units depend on state], shape (n_states, n_steps+1)
        sol.success : bool
        Use IDX dict to extract specific states: sol.y[IDX["C_ven"]]

    Raises
    ------
    ValueError
        If compound validation fails and cannot be recovered.
    RuntimeError
        If the ODE integration fails.

    Example
    -------
    >>> from simulation import run_sim
    >>> sol = run_sim(48.0, mode="healthy")
    >>> import numpy as np
    >>> from state_vector import IDX
    >>> C_ven = sol.y[IDX["C_ven"]]   # furosemide plasma [mg/L]
    """

    # -- Step 1: Build time grid ------------------------------------------------
    if n_steps is None:
        if dt is None or dt <= 0.0:
            n_steps = 10000
        else:
            n_steps = max(1, int(math.ceil(t_end_h / dt)))

    t_eval = np.linspace(0.0, t_end_h, n_steps + 1)

    # -- Step 2: Validate compounds --------------------------------------------
    # The caller must load the compound parameters for the right mode
    # (healthy or CKD-5) BEFORE calling run_sim(). run_sim() does not reload
    # them, because that would overwrite values set by the caller (for
    # example by a scenario). Only if C.drug is empty are the compounds
    # initialised here.
    _drug_already_set = isinstance(C.drug, dict) and len(C.drug) > 5
    try:
        C.validate_compounds()
    except ValueError as e:
        if _drug_already_set:
            # the caller has set the dictionaries: keep them
            print(f"[run_sim] validate_compounds note (caller dict kept): {e}")
        else:
            # nothing loaded yet
            print(f"[run_sim] C.drug empty/None - initialising for mode={mode}")
            if hasattr(C, "initialize_compounds"):
                C.initialize_compounds(mode)
            elif hasattr(C, "load_compounds"):
                C.load_compounds(mode)
            C.validate_compounds()  # raise if still failing

    # -- Step 2b: Parameter check table ---------------------------------------
    # The reference values in parameter_audit are read when that module is
    # imported. Reloading it makes sure that edits to compound_parameters.py
    # made during the same Python session are included.
    import importlib
    import parameter_audit as _pa_mod
    importlib.reload(_pa_mod)
    from parameter_audit import audit_compounds

    _phys_for_audit = make_phys(mode)
    # strict=False: the check only prints, it never stops the run.
    _audit_strict = kwargs.get("audit_strict", False)
    audit_compounds(C.drug, C.tox1, C.tox2, C.tox3, _phys_for_audit,
                    scenario=mode, strict=_audit_strict)

    # -- Step 3: Build physiology and flow chain --------------------------------
    # phys : volumes and flows for the mode; flows : tubular flow entering
    # each segment [L/h]
    phys  = make_phys(mode)
    flows = flows_L_per_h(phys)

    # -- Step 4: Build initial conditions --------------------------------------
    # When the run starts from the end of a previous (steady-state) run,
    # the cumulative excreted amounts Ae_* are set back to 0. Ae then only
    # counts what is excreted during THIS run, so the renal clearance over
    # the run is CL_R = Ae / AUC. All other states (toxin concentrations,
    # kidney compartments) are taken over from the previous run.
    if custom_initial_conds is not None:
        for key in list(custom_initial_conds.keys()):
            if key.startswith("Ae_"):
                custom_initial_conds[key] = 0.0

    y0 = initial_conditions(phys, custom_initial_conds=custom_initial_conds)
    if custom_initial_conds is not None:
        # make sure the toxin central concentrations are the given values
        for toxin_state in ["C_tox1_cen", "C_tox2_cen", "C_tox3_cen"]:
            if toxin_state in custom_initial_conds and toxin_state in IDX:
                idx = IDX[toxin_state]
                y0[idx] = custom_initial_conds[toxin_state]
                if K.VERBOSE:
                    print(f"  [initial] {toxin_state} = {y0[idx]:.4f} mg/L")
    # -- Step 5: Verbose pre-run summary ---------------------------------------
    if K.VERBOSE:
        print(f"\n=== SIMULATION START ===")
        print(
            f"  mode={mode}  |  t_end={t_end_h} h  |  n_steps={n_steps}  |  "
            f"drug.enabled={C.drug.get('enabled', False)}  |  "
            f"drug.inhibit={C.drug.get('inhibit', False)}  |  "
            f"USE_APICAL_MM={K.USE_APICAL_MM}"
        )
        print(
            f"  PHYSIOLOGY: Q_kidney={phys['Q_kidney0']:.2f} L/h  "
            f"GFR={phys['Q_GFR']:.3f} L/h  "
            f"Urine={phys['Q_urine']:.4f} L/h  "
            f"Vcen={phys['V_ven0']:.1f} L"
        )

        if custom_initial_conds:
            nonzero = {k: v for k, v in custom_initial_conds.items() if v != 0.0}
            if nonzero:
                print(f"  CUSTOM IC ({len(nonzero)} non-zero states):")
                for k, v in list(nonzero.items())[:5]:
                    print(f"    {k}: {v:.4g}")
                if len(nonzero) > 5:
                    print(f"    ... and {len(nonzero)-5} more")

        for nm, tox in [("IS", C.tox1), ("HA", C.tox2), ("pCS", C.tox3)]:
            if tox is not None:
                print(
                    f"  {nm}: C0={tox.get('C0', 0):.3f} mg/L  "
                    f"R0={tox.get('R0', 0):.2f} mg/h  "
                    f"fu_p={tox.get('fu_p', 0):.3f}  "
                    f"Vmax_baso={tox.get('Vmax_baso_uptake', 0):.1f} mg/h  "
                    f"Km={tox.get('Km_baso_uptake', 0):.3f} mg/L"
                )

        if C.drug is not None:
            d = C.drug
            print(
                f"  DRUG: enabled={d.get('enabled', False)}  "
                f"fu_p={d.get('fu_p', float('nan')):.3f}  "
                f"BP={d.get('BP', float('nan')):.2f}  "
                f"IR={d.get('IR', 0)} mg/h  "
                f"t0={d.get('t_start_h', 0)} h  "
                f"t_inf={d.get('t_inf_h', 0)} h  "
                f"k_elim={d.get('k_elim_hinv', 0):.4f} h-1  "
                f"Vperi={d.get('Vperi_L', 0)} L  "
                f"k12={d.get('k12_hinv', 0):.4f} h-1  "
                f"k21={d.get('k21_hinv', 0):.4f} h-1  "
                f"use_tissue={d.get('use_tissue', False)}"
            )

        print(
            f"  [infusion] drug.enabled={C.drug.get('enabled', False)}  "
            f"IR={C.drug.get('IR', 0)} mg/h  "
            f"t_start={C.drug.get('t_start_h', 0)} h  "
            f"t_inf={C.drug.get('t_inf_h', 0)} h  "
            f"-> infuses from {C.drug.get('t_start_h', 0)} to "
            f"{C.drug.get('t_start_h', 0) + C.drug.get('t_inf_h', 0)} h"
        )

    # -- Step 6: reset the 'print once' flags stored on kidney_rhs -----------
    try:
        from ode_system_twocompartment import kidney_rhs as _rhs
        _rhs._printed_debug = False
        _rhs._apical_guarded = False
        for attr in list(dir(_rhs)):
            if (
                attr.startswith("_printed_cin_")
                or attr.startswith("_checked_")
                or attr.startswith("_warned_zero_")
                or attr.startswith("_p1_lumen_")
                or attr in ("_flow_debug_done", "_dbg_last_bucket",
                            "_printed_peak_debug", "_tissue_dist")
            ):
                try:
                    delattr(_rhs, attr)
                except AttributeError:
                    pass
    except ImportError:
        pass

    # -- Step 7: Integrate -----------------------------------------------------
    from ode_system_twocompartment import kidney_rhs

    try:
        t_start_inf = float(C.drug.get("t_start_h", 0.0))
        t_end_inf = t_start_inf + float(C.drug.get("t_inf_h", 0.0))
        drug_active = (C.drug.get("enabled", False)
                       and float(C.drug.get("IR", 0.0)) > 0.0
                       and 0.0 < t_end_inf < t_end_h)

        if drug_active:
            # Part 1: t = 0 -> end of infusion
            n1 = max(2, int(round(n_steps * t_end_inf / t_end_h)))
            t_eval_1 = np.linspace(0.0, t_end_inf, n1 + 1)

            sol1 = solve_ivp(
                fun=lambda t, y: kidney_rhs(t, y, phys=phys, FLOWS_L_HR=flows),
                t_span=(0.0, t_end_inf),
                y0=y0,
                method="Radau",
                t_eval=t_eval_1,
                rtol=K.RTOL,
                atol=K.ATOL,
            )

            # Part 2: end of infusion -> t_end_h (no infusion)
            n2 = n_steps - n1
            t_eval_2 = np.linspace(t_end_inf, t_end_h, n2 + 1)
            y1 = sol1.y[:, -1]  # end of part 1 = start of part 2

            sol2 = solve_ivp(
                fun=lambda t, y: kidney_rhs(t, y, phys=phys, FLOWS_L_HR=flows),
                t_span=(t_end_inf, t_end_h),
                y0=y1,
                method="Radau",
                t_eval=t_eval_2,
                rtol=K.RTOL,
                atol=K.ATOL,
            )

            # join the two parts (the shared time point is kept once)
            from types import SimpleNamespace
            sol = SimpleNamespace(
                t=np.concatenate([sol1.t, sol2.t[1:]]),
                y=np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1),
                success=sol1.success and sol2.success,
                message=sol1.message if not sol1.success else sol2.message,
            )
        else:
            # no infusion stop inside the time window: one solve
            sol = solve_ivp(
                fun=lambda t, y: kidney_rhs(t, y, phys=phys, FLOWS_L_HR=flows),
                t_span=(0.0, t_end_h),
                y0=y0,
                method="Radau",
                t_eval=t_eval,
                rtol=K.RTOL,
                atol=K.ATOL,
            )
    except Exception as e:
        raise RuntimeError(
            f"ODE integration failed: {e}\n"
            "Check compound parameters and physiology values."
        ) from e

    # -- Step 8: Post-run checks -----------------------------------------------
    if not sol.success:
        print(
            f"[WARN] run_sim: solver did not converge. "
            f"Message: {sol.message}. "
            f"Results may be unreliable."
        )

    if K.VERBOSE:
        print(f"\n  Simulation completed: {len(sol.t)} timepoints, "
              f"success={sol.success}")

    # Furosemide check: did the infusion reach the central compartment?
    if C.drug.get("enabled", False) and "C_ven" in IDX:
        C_ven_max = float(np.nanmax(sol.y[IDX["C_ven"]]))
        if C_ven_max <= 1e-12:
            print(
                "\n[WARN] Drug enabled but C_ven stayed ~0 throughout. "
                "Check drug['IR'] > 0, drug['t_start_h'], drug['t_inf_h'] "
                "and that drug['enabled'] = True was set before run_sim()."
            )
        else:
            print(
                f"\n[OK] Drug reached peak plasma: "
                f"{C_ven_max:.4e} mg/L at t="
                f"{sol.t[np.nanargmax(sol.y[IDX['C_ven']])]:.2f} h"
            )

    return sol


# ---------------------------------------------------------------------------
# Helper to read the solution
# ---------------------------------------------------------------------------

def extract_results(sol, compound: str = "all") -> dict:
    """
    Extract named concentration and excretion time-series from a solution.

    Same as sol.y[IDX[...]], but returns a dictionary with readable names.
    All arrays have the same length as sol.t.

    Parameters
    ----------
    sol      : OdeResult from run_sim()
    compound : "all", "drug", "IS", "HA", or "pCS"

    Returns
    -------
    dict with keys:
      "time"          : time vector [h]
      "plasma_drug"   : C_ven [mg/L]          (if compound in ["all","drug"])
      "tissue_drug"   : C_peri [mg/L]         (if compound in ["all","drug"])
      "bladder_drug"  : C_bladder [mg/L]      (if compound in ["all","drug"])
      "excreted_drug" : Ae_drg [mg]           (if compound in ["all","drug"])
      "plasma_IS"     : C_tox1_cen [mg/L]     (if compound in ["all","IS"])
      "excreted_IS"   : Ae_t1 [mg]            (if compound in ["all","IS"])
      "plasma_HA"     : C_tox2_cen [mg/L]     (if compound in ["all","HA"])
      "excreted_HA"   : Ae_t2 [mg]            (if compound in ["all","HA"])
      "plasma_pCS"    : C_tox3_cen [mg/L]     (if compound in ["all","pCS"])
      "excreted_pCS"  : Ae_t3 [mg]            (if compound in ["all","pCS"])
    """
    t = sol.t
    y = sol.y
    results: dict = {"time": t}

    if compound in ("all", "drug"):
        if "C_ven"     in IDX: results["plasma_drug"]   = y[IDX["C_ven"]]
        if "C_peri"    in IDX: results["tissue_drug"]   = y[IDX["C_peri"]]
        if "C_bladder" in IDX: results["bladder_drug"]  = y[IDX["C_bladder"]]
        if "Ae_drg"    in IDX: results["excreted_drug"] = y[IDX["Ae_drg"]]

    if compound in ("all", "IS"):
        if "C_tox1_cen" in IDX: results["plasma_IS"]   = y[IDX["C_tox1_cen"]]
        if "Ae_t1"      in IDX: results["excreted_IS"] = y[IDX["Ae_t1"]]

    if compound in ("all", "HA"):
        if "C_tox2_cen" in IDX: results["plasma_HA"]   = y[IDX["C_tox2_cen"]]
        if "Ae_t2"      in IDX: results["excreted_HA"] = y[IDX["Ae_t2"]]

    if compound in ("all", "pCS"):
        if "C_tox3_cen" in IDX: results["plasma_pCS"]   = y[IDX["C_tox3_cen"]]
        if "Ae_t3"      in IDX: results["excreted_pCS"] = y[IDX["Ae_t3"]]

    return results
