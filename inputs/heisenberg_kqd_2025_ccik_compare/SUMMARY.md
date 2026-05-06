# CCIK vs Nature 2025 KQD

CCIK does not beat experimental KQD at the same D.

Pointwise wins: 17/30 valid case-dimension comparisons.

## Setup

- Thick-restart style parameters: `m_cycle=10`, `ncycles=1`, `nroot=3`
- Compression ladder: `[1000, 3000, 10000, 30000, 100000]` with `Kprop = Kstore`
- Paper KQD curve: reconstructed from the published companion notebook's Toeplitz `H~` and `S~` first rows.
- Local reference: exact diagonalization in the fixed-k sector when tractable; otherwise the published exact value from the companion notebook.

## Case Notes

- `N = 56, k = 1`
  initial state: `[1]` via `fallback_degree_match`
  exact reference: `47.322497152132` (`local_exact`)
  best final-D Kstore: `56` with abs error `0.231454`
  paper final-D abs error: `2.23881`
- `N = 44, k = 3`
  initial state: `[7, 29, 33]` via `seed_from_S_meas`
  exact reference: `19.665326355477` (`local_exact`)
  best final-D Kstore: `10000` with abs error `0.309967`
  paper final-D abs error: `3.23764`
- `N = 42, k = 5`
  initial state: `[1, 22, 23, 28, 38]` via `screened_override_best_D10`
  exact reference: `0.214373949644` (`published_exact`)
  best final-D Kstore: `100000` with abs error `3.42387`
  paper final-D abs error: `2.38388`

## Assumptions

- The benchmark uses the published qubit Hamiltonians from the paper's companion notebooks, not the molecular QP path.
- The local CCIK curve is a self-contained compressed Krylov approximation on the same Heisenberg Hamiltonians.
- For the largest `N=42, k=5` sector, the exact reference used for the error metric is the published exact value from the companion notebook.
