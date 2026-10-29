# Guide to the code

Author: Leyla Noroozbabaee, MaCSBio, Maastricht University, October 2026

This guide explains how the code is organised and how a simulation runs.
Read it next to the code.

## 1. The question behind the model

Furosemide is a loop diuretic that is often given to patients with CKD.
It reaches its site of action, the lumen of the loop of Henle, by
tubular secretion: OAT1 takes it up from the blood into the proximal
tubule cell, and apical transporters move it into the lumen. The uremic
toxins IS, HA and pCS are removed by the same route. Furosemide can
therefore compete with the toxins for OAT1 and lower their renal
clearance. The model quantifies this effect in healthy subjects and in
CKD stage 5.

## 2. Suggested reading order

1. `README.md`
2. `src/ode_system_twocompartment.py`: the header explains all
   compartments and processes. Then read `kidney_rhs()` step by step.
3. `src/compound_parameters.py`: parameter values and their sources.
4. `src/constants.py` and `src/physiology.py`: flows and volumes.
5. `src/scenario_manager.py` and `src/simulation.py`: how a run is set
   up and solved.
6. Section 3 below: a complete example from steady state to scenario.

## 3. How one simulation works

```
set_scenario(...)            scenario_manager.py
   -> initialize_compounds   compounds.py  (reads compound_parameters.py)
   -> drug["enabled"], drug["inhibit"]
run_sim(t_end, mode, ...)    simulation.py
   -> make_phys(mode)        physiology.py (reads constants.py)
   -> initial_conditions()   state_vector.py
   -> solve_ivp(kidney_rhs)  ode_system_twocompartment.py
returns sol: sol.t (time), sol.y (all states)
```

A typical study has two steps:

1. Steady state: a long run without furosemide (hundreds of hours), so
   that the toxin concentrations no longer change.
2. Scenario: a short run (for example 12 h) that starts from the end of
   step 1, with or without furosemide and with or without inhibition.

```python
import constants as K
K.VERBOSE = False                      # less printed output

from scenario_manager import set_scenario
from simulation import run_sim
from state_vector import IDX

set_scenario("CKD_NO_DRUG")
sol_ss = run_sim(300.0, mode="ckd5", dt=1.0)
ic = {name: sol_ss.y[i, -1] for name, i in IDX.items()}

set_scenario("CKD_WITH_DRUG")
sol = run_sim(12.0, mode="ckd5", custom_initial_conds=ic, dt=0.05)

C_IS = sol.y[IDX["C_tox1_cen"]]        # IS in the central compartment [mg/L]
```

Run this from the `src/` folder, or add `src/` to `sys.path`.

## 4. Key equations

All states are total concentrations. Transporters and passive diffusion
see the unbound concentration:

    blood: Cu = C * fu_p / BP          cell: Cu = C * fu_cell

OAT1 uptake in a proximal segment, with competitive inhibition:

    J = w * Vmax * Cu_tox / (Km * (1 + Cu_furo / Ki) + Cu_tox)

w is the share of the segment (P1 0.4, P2 0.4, P3 0.2).

Central compartment of a toxin:

    dC/dt = R0 / Vcen - (Qkid / Vcen) * (C - C_renal_vein)

At steady state the renal clearance equals R0 / C, so a lower clearance
gives a higher concentration.

Renal clearance from a simulation:

    CL_R = Ae / AUC

Ae is the amount excreted in urine over a time window and AUC the area
under the concentration curve over the same window.

## 5. Healthy versus CKD-5

| | healthy | CKD-5 |
|---|---|---|
| GFR | 120 mL/min | 12 mL/min |
| renal blood flow | 1200 mL/min | 420 mL/min |
| urine flow | 1.0 mL/min | 0.5 mL/min |
| OAT1 (basolateral) Vmax | 100 % | 5 % |
| apical Vmax | 100 % | 15 % |
| fu_p IS / HA / pCS | 0.16 / 0.66 / 0.06 | 0.10 / 0.52 / 0.05 |
| fu_p furosemide | 0.02 | 0.05 |

Toxin production (R0), Km, Ki and the non-renal elimination of
furosemide are the same in both states.

## 6. Things to keep in mind

- Use the same `mode` in `set_scenario()` and `run_sim()`.
- `set_scenario()` reloads all parameters. Change a parameter AFTER
  calling it, otherwise your change is lost.
- `K.VERBOSE = True` prints a lot of intermediate values. This is useful
  for checking but slow for long runs.
- `parameter_audit` prints a table before each run. WARN lines only mean
  that a value differs from the paper set; they are expected when you
  change a parameter on purpose.
- Results go to `results/`. Cache files there are reused; delete them if
  you changed the model and want new results.

## 7. Points that are still open

These are known points to look at in follow-up work:

- `CL_int_cell` of furosemide (1.3e-4 L/h) is still active in
  `compound_parameters.py`.
- The non-renal clearance of furosemide in the model is
  k_elim x Vcen = 0.12 x 5 = 0.6 L/h, in healthy and CKD-5.
- The reference values in the bar charts of the comparison scripts
  (`LIT_CONC_CKD`, `LIT_CLEAR_CKD`, `lit_conc`, `lit_clear`, `CLIN_MEAN`)
  should be checked against the literature table of the paper.
- `plot_clearance_flux_decomposition.py` estimates the fluxes after the
  run with simplified equations (see the docstring of
  `compute_fluxes_for_scenario_drug`).
- The healthy comparison script converts blood to plasma concentrations
  (divide by BP); the CKD-5 script does not.
