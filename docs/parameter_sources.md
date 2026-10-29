# Parameter sources

Author: Leyla Noroozbabaee, MaCSBio, Maastricht University, October 2026

The values below are those in `src/compound_parameters.py` and
`src/constants.py`. If this table and the code ever differ, the code is
what the model uses.

## OAT1 uptake: Km

| Compound | Km (uM) | Km (mg/L) | Source |
|---|---|---|---|
| Furosemide | 38.9 | 12.87 | Hasannejad et al. 2004 |
| Indoxyl sulfate (IS) | 18 | 3.84 | Chang et al. 2023 |
| Hippuric acid (HA) | 24 | 4.302 | Chang et al. 2023 |
| p-Cresyl sulfate (pCS) | 120 | 22.58 | Chang et al. 2023 |

Km [mg/L] = Km [uM] x MW [g/mol] / 1000 (MW in `constants.py`).

## Inhibition constant Ki

Ki = 0.35 mg/L for IS, HA and pCS. Selected from a per-toxin sensitivity
analysis under healthy conditions (about 5 % average increase of the
toxin concentrations, chosen by visual inspection, no formal fitting).
Values above about 1 mg/L gave little or no interaction. The clinical
data of Costes-Albrespic et al. 2025 were used afterwards only as a
qualitative comparison.

## Plasma protein binding (fu_p)

| Compound | healthy | CKD-5 | Source |
|---|---|---|---|
| Furosemide | 0.02 | 0.05 | Rane et al. 1978 (higher unbound fraction in uremia) |
| IS | 0.16 | 0.10 | Deltombe et al. 2015 |
| HA | 0.66 | 0.52 | Deltombe et al. 2015 |
| pCS | 0.06 | 0.05 | Deltombe et al. 2015 |

## Proximal tubule structure

| Item | Value | Source |
|---|---|---|
| model framework | segments, compartments, passive diffusion | Huang & Isoherranen 2018, 2020 |
| basolateral area | apical area / 30 | Huang & Isoherranen 2018 |
| OAT1 share P1 / P2 / P3 | 0.4 / 0.4 / 0.2 | Breljak et al. 2016 |

## Physiology

| Item | healthy | CKD-5 |
|---|---|---|
| GFR | 120 mL/min | 12 mL/min (x 0.10; CKD stage 5 = GFR < 15 mL/min, KDIGO 2012) |
| renal blood flow | 1200 mL/min | 420 mL/min (x 0.35, model assumption) |
| urine flow | 1.0 mL/min | 0.5 mL/min (model assumption) |
| central volume | 5 L | 5 L |

## Furosemide

| Parameter | Value | Note |
|---|---|---|
| Vss | 5.33 L | Vcen x (1 + k12/k21) = 5 x (1 + 0.033/0.5) |
| k_elim | 0.12 1/h | non-renal elimination, same in healthy and CKD-5 |
| default dose | 120 mg IV over 2 h | |
| oral bioavailability F | 0.60 | Lasix product label (oral scripts only) |
| absorption rate ka | 1.5 1/h | Hammarlund et al. 1984 (oral scripts only) |

## Clinical comparison

Costes-Albrespic et al. 2025: patients prescribed > 120 mg/day
furosemide, compared with patients without furosemide (adjusted):

| Toxin (free concentration) | difference | 95 % CI |
|---|---|---|
| IS | +31.9 % | 13.5 to 53.3 % |
| pCS | +29.3 % | 6.1 to 57.8 % |

## References

- Breljak D, et al. Am J Physiol Renal Physiol 2016;311:F227.
- Chang SY, Huang W, Chapron A, et al. Pharm Res 2023;40:2597-2606.
- Costes-Albrespic M, Alencar de Pinho N, Larabi IA, et al. Kidney Int Rep 2025;10:2165-2177.
- Deltombe O, Van Biesen W, Glorieux G, Eloot S. Toxins 2015;7:3933-3946.
- Hammarlund MM, Paalzow LK, Odlind B. Eur J Clin Pharmacol 1984;26:197.
- Hasannejad H, Takeda M, Taki K, et al. J Pharmacol Exp Ther 2004;308:1021-1029.
- Huang W, Isoherranen N. CPT Pharmacometrics Syst Pharmacol 2018;7:593-602.
- Huang W, Isoherranen N. CPT Pharmacometrics Syst Pharmacol 2020;9:571-583.
- KDIGO 2012 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. Kidney Int Suppl 2013;3:1-150.
- Rane A, Villeneuve JP, Stone WJ, et al. Clin Pharmacol Ther 1978;24:199-207.
