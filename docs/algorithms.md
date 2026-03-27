# Algorithms

The active codebase implements three sparse selected-space compressed-Krylov solvers over a CAS Hamiltonian:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

All three operate in a determinant basis built from the CAS Hamiltonian returned by `ccik.pyscf_cas`.

## Shared building blocks

Each solver repeatedly works with these objects:

- `q_k`: the current sparse Krylov vector
- `E_k = <q_k | H | q_k>`: Rayleigh quotient for the current vector
- `H_II`: diagonal Hamiltonian elements on the selected support
- `score_I = |(v_proj)_I|^2 / |E_k - H_II|`: determinant ranking score on the selected space
- `S_k`: selected support used to approximate `H q_k` and compress the next basis vector

Because the code compresses vectors explicitly, the Ritz solve uses a generalized eigenproblem `H c = E S c` rather than assuming exact orthonormality.

## Selector modes

The selected support is built by a selector backend:

- `selector_mode = "cipsi"`
  - default for `ccik` and `ccik_thick`
  - uses PySCF selected-CI expansion
- `selector_mode = "stochastic"`
  - default for `ccik_stochastic`
  - uses fixed-budget stochastic discovery and repeated projected contractions
- `selector_mode = "full"`
  - explicit validation fallback only
  - forms the full dense `H q_k`

## `ccik`

Implementation: `src/ccik/core.py`

Entry point:

- `ccik_ground_energy_dense(...)`

Workflow:

1. Start from the Hartree-Fock determinant.
2. Build a selected determinant space for the current sparse Krylov vector.
3. Contract `H q_k` only on that selected space.
4. Rank external determinants with the score above.
5. Keep the selected determinants plus a `Kv`-sized stabilizer from the largest projected amplitudes.
6. Orthogonalize and compress the next Krylov vector.
7. Solve the generalized Ritz problem.

Important knobs:

- `m`: total Krylov dimension
- `nadd`: new determinants added per step
- `nkeep`: support retained in each basis vector
- `Kv`: projected-amplitude stabilizer size
- `selector_mode`, `selector_depth`: selector backend and depth

## `ccik_thick`

Implementation: `src/ccik/thick_restart.py`

Entry point:

- `ccik_ground_energy_dense_thick_restart(...)`

Difference from baseline:

- the solver builds a cycle of length `m_cycle`
- solves Ritz at the end of the cycle
- keeps the best `nroot` Ritz vectors
- starts the next cycle from those vectors

Important knobs:

- `m_cycle`: local Krylov dimension per cycle
- `ncycles`: restart budget
- `nroot`: number of retained Ritz vectors
- `tol`: stopping threshold between successive cycle energies

## `ccik_stochastic`

Implementation: `src/ccik/stochastic.py`

Entry point:

- `ccik_ground_energy_stochastic(...)`

Difference from baseline:

- candidate discovery changes from deterministic selected-CI expansion to stochastic excitation proposals
- `selector_depth` is real here: each depth performs a fixed-budget discovery round, accumulates the visited support, and recomputes a projected contraction on that support
- score, compression, orthogonalization, and Ritz extraction remain the same as in the other solvers

The stochastic discovery stage uses:

- `n_walkers`: proposals per depth
- `parent_power`: bias toward large-amplitude parent determinants
- `p_double`: single vs double excitation split
- `mixed_double_weight`: weighting of mixed-spin doubles
- `seed`: reproducibility control

## Current benchmark takeaways

From `experiments/sparse_backend_benchmark_20260312/SUMMARY.md`:

- On N2 / 6-31G / CAS(10,10), tuned `ccik` and tuned `ccik_stochastic` both stayed below `1e-3 Eh` on the tested validation set.
- Tuned `ccik_thick` remained slightly above `1e-3 Eh` at the short bond.
- For deterministic solvers, `selector_depth = 2` was consistently harmful on the tested hard points.
- For stochastic, `selector_depth = 2` was necessary at the tuned walker budget.
- The tuned CAS(10,10) settings do not transfer cleanly to stretched CAS(12,12); sparse storage helps, but the harder multireference regime still needs a better selector/refinement strategy.
