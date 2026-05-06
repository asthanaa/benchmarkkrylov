# High-Accuracy CCIK Refinement on 49Q_v1

A sub-1e-5 solution was obtained with CCIK-guided pool refinement.

Budget plan:
- Requested total determinant allowance: `10000000`
- Planned restart interpretation: with `nroot = 5`, a full-budget restart cap would allow `kstore = 2000000` per retained Ritz vector.
- Actual converged pool size: `130013` (1.3001% of the allowed budget)

Baseline CCIK run:
- Krylov dimension `17`, `Kprop = 1000`, `Kstore = 1000`, `nroot = 5`, `ncycles = 16`

Final result:
- Exact reference energy: `-85.586625066132`
- Final refined energy: `-85.586625069864`
- Final absolute error: `3.732438358384e-09`

Stages:
- baseline_variational: pool `8522`, energy `-85.388810140401`, error `1.978149257308e-01`
- exactized_pool0: pool `8522`, energy `-85.460458765904`, error `1.261663002275e-01`
- exactized_pool1: pool `74539`, energy `-85.576625069864`, error `9.999996267069e-03`
- exactized_pool2: pool `130013`, energy `-85.586625069864`, error `3.732438358384e-09`

Plot note:
- `ccik_high_accuracy_refinement.*` now overlays the paper experimental SKQD points from `data/paper_experimental_skqd.csv`.
- The paper's digitized zero-error endpoints are clipped to the plot floor because the figure uses a log-scale error axis.

Artifacts:
- `data/high_accuracy_refinement.csv`
- `data/high_accuracy_budget_plan.json`
- `data/paper_experimental_skqd.csv`
- `ccik_high_accuracy_refinement.png`
- `ccik_high_accuracy_refinement.pdf`
