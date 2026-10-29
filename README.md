# Kidney PBPK model: furosemide and protein-bound uremic toxins

Author: Leyla Noroozbabaee
Maastricht Centre for Systems Biology (MaCSBio), Faculty of Science and
Engineering, Maastricht University, Maastricht, The Netherlands
October 2026

This repository contains the Python code of a mechanistic kidney
physiologically based pharmacokinetic (PBPK) model. The model describes
how furosemide competitively inhibits the OAT1-mediated tubular secretion
of three protein-bound uremic toxins, indoxyl sulfate (IS), hippuric acid
(HA) and p-cresyl sulfate (pCS), in healthy subjects and in chronic
kidney disease stage 5 (CKD-5).

It is the code of the paper "A Mechanistic Kidney PBPK Model to Quantify
Transporter-Mediated Furosemide-Toxin Interactions in CKD".

New to the code? Start with `GUIDE.md`.

## Model in short

- Central (blood) compartment for furosemide and for each toxin;
  furosemide also has a peripheral tissue compartment.
- The kidney is 11 nephron segments in series: proximal tubule (P1, P2,
  P3), loop of Henle (LH1, LH2), distal tubule (DT) and collecting duct
  (CDC1-CDC5). Each segment has a vascular, a cell and a lumen
  compartment (142 states in total).
- Processes: glomerular filtration of the unbound compound, OAT1 uptake
  and apical secretion in the proximal tubule (Michaelis-Menten),
  passive diffusion of the un-ionised unbound form, tubular flow.
- Inhibition: furosemide raises the apparent Km of OAT1 for each toxin,
  `Km_eff = Km * (1 + Cu_furosemide / Ki)`, with Ki = 0.35 mg/L.
- The ODEs are solved with `scipy.integrate.solve_ivp` (Radau).

## Folder structure

```
src/                    model code
analysis/               scripts that run simulations and make figures
analysis/sensitivity/   one-parameter sensitivity analyses
figures/                scripts that draw paper figures from saved results
docs/                   parameter sources
results/                output of all scripts (empty in the repository)
```

## Installation and a first run

Python 3.10 or newer.

```bash
pip install -r requirements.txt
python analysis/plot_healthy_lit_comparison_3scenarios.py
```

Every script finds `src/` by itself, so it can be started from any
folder. All output (figures, CSV files, cache files) is written to
`results/`.

Run time: a single scenario takes seconds to a few minutes. The heatmap
and sensitivity scripts run the model many times and can take hours;
they save their results in a cache file and reuse it.

## Files in `src/`

| File | Content |
|---|---|
| `constants.py` | segments, membrane areas, tubular flows, CKD-5 scale factors, pH, molecular weights, solver tolerances |
| `physiology.py` | volumes and flows for healthy and CKD-5 (`make_phys`) |
| `compound_parameters.py` | all compound parameters, with sources |
| `compounds.py` | active parameter set used during a run |
| `passive_diffusion.py` | passive diffusion clearances per segment |
| `helpers.py` | unit conversion, ionisation, membrane areas |
| `state_vector.py` | names and order of the 142 states, initial values |
| `ode_system_twocompartment.py` | the ODE system (`kidney_rhs`) |
| `simulation.py` | `run_sim()`: solves the ODEs |
| `scenario_manager.py` | named scenarios, e.g. `set_scenario("CKD_WITH_DRUG")` |
| `parameter_audit.py` | prints a check of the parameters before each run |

## Scripts in `analysis/` and `figures/`

| Script | Result |
|---|---|
| `analysis/plot_healthy_lit_comparison_3scenarios.py` | healthy: no furosemide / furosemide without interaction / with interaction |
| `analysis/plot_ckd_lit_comparison_3scenarios.py` | the same for CKD-5 |
| `analysis/plot_furosemide_pk_ckd_vs_healthy.py` | furosemide PK, healthy versus CKD-5 |
| `analysis/plot_clearance_flux_decomposition.py` | share of filtration, passive diffusion, OAT1 uptake and secretion |
| `analysis/dose_response_tid_vs_qd.py` | 30 days of oral furosemide in CKD-5, three times daily versus once daily |
| `analysis/heatmap_ki_vmax_sensitivity_ode.py` | Ki x OAT1 capacity in CKD-5 |
| `analysis/heatmap_fu_sensitivity_ode.py` | unbound fraction of furosemide x unbound fraction of the toxins |
| `analysis/sensitivity/*.py` | sensitivity to Ki, fu_p, dose and R0 |
| `figures/make_three_way_comparison_figure.py` | model vs analytical estimate vs clinical data |
| `figures/heatmap_ki_vmax_from_cache.py` | Ki x Vmax figure from the saved grid |

## Choice of Ki

Ki = 0.35 mg/L is used for all three toxins. It was selected from a
sensitivity analysis per toxin under healthy conditions: this value gives
an average increase of about 5 % in the toxin concentrations, chosen by
visual inspection (no formal fitting). Values above about 1 mg/L gave
little or no interaction. The clinical study of Costes-Albrespic et al.
(Kidney Int Rep 2025;10:2165-2177) was used afterwards only as a
qualitative comparison; Ki was not fitted to it.

## Citation

Please cite the paper when you use this code (see `CITATION.cff`).

## License

MIT, see `LICENSE`. The model is a research tool and is not meant for
clinical decisions or dosing.
