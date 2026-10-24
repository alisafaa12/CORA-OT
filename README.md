# CORA-OT: reference implementation and synthetic evaluation

This repository accompanies the paper:

> A. S. Sadiq, "CORA-OT: A Multi-Stakeholder, Crowd-Augmented and LLM-Assisted Cyber Risk
> Assessment Framework for Converged IT/OT Energy Infrastructure", submitted to *Computers* (MDPI), 2026.

It contains a Python implementation of the CORA-OT formal model, the synthetic
distribution-network case used in Section 8, a separate ground-truth simulator, and
all scripts and results needed to reproduce the tables and figures of Section 8.

**All data are synthetic.** No operational, personal or real operator data are used.

## Contents

| File | Purpose |
|---|---|
| `code/cora_model.py` | Equations (2)-(16): noisy-OR propagation with AND groups, baseline probability, evidence weights, freshness, clustering, evidence update, value-based consequence with safety floor, cascading consequence, control effectiveness, residual risk, exact and greedy control selection. |
| `code/case_study.py` | Synthetic case: 20 nodes, 4 owners, attack and consequence layers, 8 scenarios, 12 controls, limits and governance parameters. |
| `code/worked_example.py` | Worked example (Sections 8.2-8.4): Table 10, Table 11, rank stability, 12-week evidence timeline, budget frontier. |
| `code/experiments.py` | Synthetic evaluation (Sections 8.5-8.9): ground-truth simulator, method comparison and ablations (E1), noise sensitivity (E2), calibration (E3), scalability (E4), deterministic LLM output gate (E5). |
| `code/make_figures.py` | Builds Figures 2-7 from `code/results/`. |
| `code/results/` | JSON and NumPy outputs used in the paper. |
| `figures/` | Figures as used in the paper. |

## Reproduce

```bash
pip install -r requirements.txt
cd code
python worked_example.py     # about 10 s
python experiments.py        # about 1 min
python make_figures.py       # writes ../figures/*.pdf
```

Random seeds are fixed (`20261019` for the experiments, `11` for rank stability, `7` for the LLM gate test).
Small differences in the last digit can occur across NumPy versions.

## Notes on interpretation

* The ground-truth simulator uses live-edge Monte Carlo cascades and monetary losses, while
  CORA-OT uses the mean-field noisy-OR rule and a value function. Both share the same graph
  structure, so missing or wrong arcs are not tested.
* FAIR-style and BAG-style baselines are simplified re-implementations of the core logic of each
  method, not full tools.
* No LLM is called. Experiment E5 tests only the deterministic output gate on synthetic outputs.

See Section 8.10 of the paper for what the synthetic evaluation does and does not show.

## Licence

MIT (code) and CC BY 4.0 (synthetic data and figures), unless stated otherwise by the author.
