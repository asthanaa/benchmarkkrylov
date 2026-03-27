# Algorithms

The active codebase implements three sparse selected-space compressed-Krylov solvers over a CAS Hamiltonian:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

All three share the same high-level Krylov structure:

- `q_k`: current sparse Krylov basis vector
- `E_k = <q_k | H | q_k>`: Rayleigh quotient
- `score_I = |(v_proj)_I|^2 / |E_k - H_II|`: determinant ranking score on the selected space
- `v_proj \approx P_{S_k} H q_k`: projected Hamiltonian action on a selected determinant space
- generalized Ritz solve because compressed vectors are not assumed to stay exactly orthonormal

## Selector modes

The selected space used to approximate `H q_k` is built by a selector backend:

- `selector_mode = "cipsi"`
  - default for `ccik` and `ccik_thick`
  - uses PySCF selected-CI expansion to build a selected determinant space
- `selector_mode = "stochastic"`
  - default for `ccik_stochastic`
  - uses fixed-budget stochastic discovery and repeated projected contractions
- `selector_mode = "full"`
  - explicit validation fallback only
  - forms the full dense `H q_k`

## `ccik`

Implementation: `src/ccik/core.py`

Workflow:

1. Start from the Hartree-Fock determinant.
2. Build a selected determinant space for the current sparse Krylov vector.
3. Contract `H q_k` only on that selected space.
4. Rank external determinants with the usual score.
5. Keep the selected determinants plus a `Kv` stabilizer from the largest projected amplitudes.
6. Orthogonalize, compress, and append the next sparse Krylov vector.
7. Solve the generalized Ritz problem.

Important parameters:

- `m`: total Krylov dimension
- `nadd`: new determinants added per step
- `nkeep`: support retained in each basis vector
- `Kv`: stabilizer size
- `selector_mode`, `selector_depth`: selector backend and depth

## `ccik_thick`

Implementation: `src/ccik/thick_restart.py`

Difference from baseline:

- build a local Krylov cycle of length `m_cycle`
- solve Ritz at the end of the cycle
- keep the best `nroot` Ritz vectors
- restart for up to `ncycles`

Important parameters:

- `m_cycle`: local Krylov dimension per restart cycle
- `ncycles`: maximum restart cycles
- `nroot`: retained Ritz vectors per restart
- `tol`: convergence threshold between restart energies
- `nadd`, `nkeep`, `Kv`, `selector_mode`, `selector_depth`: same meanings as in `ccik`

## `ccik_stochastic`

Implementation: `src/ccik/stochastic.py`

Difference from baseline:

- candidate discovery is stochastic rather than deterministic selected-CI expansion
- `selector_depth` is real in this solver: each depth performs a fixed-budget discovery round, accumulates the discovered support, and recomputes a projected contraction on that accumulated support
- the final Krylov update still uses the same score, compression, orthogonalization, and Ritz machinery as the other solvers

Important parameters:

- `m`, `nadd`, `nkeep`, `Kv`: same meanings as in `ccik`
- `n_walkers`: stochastic proposals per depth
- `selector_depth`: number of discovery / projection rounds
- `parent_power`: parent sampling bias
- `p_double`: single vs double excitation split
- `mixed_double_weight`: mixed-spin double weighting
- `seed`: reproducibility control
- `eps_denom`: denominator floor in the score

## Current benchmark takeaways

From `experiments/sparse_backend_benchmark_20260312/SUMMARY.md`:

- On N2 / 6-31G / CAS(10,10), tuned `ccik` and tuned `ccik_stochastic` both stayed below `1e-3 Eh` on the tested validation set.
- Tuned `ccik_thick` remained slightly above `1e-3 Eh` at the short bond.
- For deterministic solvers, `selector_depth = 2` was consistently harmful on the tested hard points.
- For stochastic, `selector_depth = 2` was necessary at the tuned walker budget.
- The tuned CAS(10,10) settings do not transfer to stretched CAS(12,12); sparse storage helps, but the harder multireference regime still needs a better selector/refinement strategy.
