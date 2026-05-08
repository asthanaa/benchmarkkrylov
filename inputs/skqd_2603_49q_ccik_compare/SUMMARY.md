# CCIK vs experimental SKQD on 49Q_v1

CCIK beats the paper's experimental SKQD at the same projected configuration-pool size.

Benchmark facts:
- Paper: arXiv:2603.03496
- Public Hamiltonian: `49Q_v1_hamiltonian.npy` with 419 Pauli terms
- Public guiding state: `49Q_v1_initial_state_index.npy`
- Exact reference energy for the public Hamiltonian: `-85.586625066132`
- Reference method: recover the global identity shift omitted from the public Pauli list by matching the public guiding-state diagonal energy to the paper's known local row-0 diagonal construction.
- CCIK x-axis proxy: union support size of the retained sparse Krylov basis, used as the projected configuration-pool dimension.

First win against experimental SKQD:
- Largest projected configuration-pool dimension: `2051`
- Krylov dimension at that win: `17`
- CCIK absolute error: `1.978149e-01`
- Paper experimental SKQD absolute error used for comparison: `2.426652e-01` at pool size `244296` (paper point `1`)
- Paper/CCIK projected-pool size ratio at that win: `119.11x`
- Settings: mode=`strict_restart`, Kprop=`1000`, Kstore=`1000`, nroot=`5`, ncycles=`16`

Best searched Krylov-dimension-17 comparison:
- Best CCIK error: `1.978149e-01`
- Same-or-larger paper SKQD point used for comparison: `2.426652e-01` at pool size `244296`
- Best CCIK settings at Krylov dimension 17: mode=`strict_restart`, Kprop=`1000`, Kstore=`1000`, nroot=`5`, ncycles=`16`
- Corresponding projected configuration-pool dimension: `2051`

Artifacts:
- `paper_experimental_skqd.csv`
- `ccik_runs.csv`
- `ccik_best_frontier.csv`
- `exact_reference.json`
- `ccik_vs_experimental_skqd.png`
- `ccik_vs_experimental_skqd.pdf`
