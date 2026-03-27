# `ccik`

This package implements determinant-sparse compressed Krylov solvers for CAS
Hamiltonians.

Current solver variants:

- baseline dense CCIK
- thick-restart CCIK
- stochastic-discovery CCIK

## Current algorithmic defaults

The maintained propagation path is matvec-first, not PT2-first.

For each sparse Krylov vector `q_k`, the code now:

1. builds a candidate determinant pool
2. evaluates a projected sparse approximation to `H q_k`
3. ranks candidates by projected-matvec amplitude or Arnoldi-residual amplitude
4. orthogonalizes against the existing basis
5. compresses only after orthogonalization
6. solves the generalized Ritz problem in the compressed basis

The default selector is residual-driven:

- `selector_mode = "abs_residual"`

Legacy PT2-like ranking is still available only for ablation/backward
compatibility:

- `selector_mode = "pt2_legacy"`

## Compression policies

Three compression policies are implemented.

### `compression_policy = "topk"`

This is the legacy size-capped mode.

Main knobs:

- `nadd`
- `nkeep`
- `Kv`
- `generator_keep`
- `tmp_keep_factor`
- `preserve_prev_support`
- `preserve_prev_support_cap`

### `compression_policy = "threshold"`

This is the accuracy-oriented threshold mode.

User-facing thresholds:

- `store_abs_tol`
  - drop normalized residual coefficients below this threshold when storing
    the next Krylov vector
- `propagate_abs_tol`
  - drop normalized `q_k` coefficients below this threshold when deciding which
    determinants participate in the next `H q_k` propagation

Threshold semantics:

- thresholds act on normalized Krylov vectors
- if `propagate_abs_tol` removes the entire vector, the largest coefficient is
  kept as a safeguard
- if `store_abs_tol` removes the entire orthogonalized residual, the largest
  residual coefficient is kept as a safeguard
- threshold mode does not silently fall back to top-k compression
- if threshold mode exceeds internal candidate or stored-support safety limits,
  the code raises a runtime error with an explicit diagnostic

In threshold mode, the active path is:

- propagation support from `|q_{k,J}| >= propagate_abs_tol`
- candidate generation from connected singles/doubles off that support, or full
  mode where available
- storage from `|r_{k,I}| >= store_abs_tol` after orthogonalization

### `compression_policy = "fixed_budget"`

This is the recommended large-space sparse policy.

User-facing budgets:

- `fixed_store_keep`
  - keep only the top-`Kstore` coefficients by absolute value when storing
    Krylov, restart, and final Ritz vectors
- `fixed_propagate_keep`
  - propagate only from the top-`Kprop` coefficients of the current normalized
    stored vector

Fixed-budget semantics:

- no threshold comparisons are used in the Krylov update path
- propagation support is the top `min(Kprop, nnz(q_k))` determinants of `q_k`
- projected `H q_k` workspaces are compressed to the top `Kstore` amplitudes
- after orthogonalization, the next Krylov vector is compressed to the top
  `Kstore` residual coefficients and then normalized
- thick-restart retained vectors and the final lowest Ritz vector are also
  compressed to top `Kstore`

In fixed-budget mode, the active path is:

- propagation support from the top `fixed_propagate_keep` `|q_{k,J}|` entries
- candidate generation from connected singles/doubles off that propagation
  support, or full mode where available
- storage from the top `fixed_store_keep` amplitudes after orthogonalization

## Core modules

- `core.py`
  - baseline dense CCIK solver
- `thick_restart.py`
  - thick-restart CCIK solver
- `stochastic.py`
  - stochastic candidate-discovery solver
- `propagation.py`
  - shared projected-matvec, orthogonalization, compression, and diagnostics
- `selection.py`
  - candidate-pool construction and projected `H q_k` workspaces
- `sparse_ci.py`
  - sparse determinant-vector operations and threshold pruning helpers
- `reference_debug.py`
  - Python oracle/debug path used for exact step-by-step parity checks

## Key parameters

Structural solver controls:

- dense: `m`
- thick restart: `m_cycle`, `ncycles`, `nroot`, `tol`

Selector/candidate controls:

- `candidate_mode = "selected_ci" | "connected_sd" | "stochastic" | "full"`
- `selector_mode = "abs_hq" | "sq_hq" | "abs_residual" | "sq_residual" | "pt2_legacy"`
- `include_current_support`
- `selector_depth`

Validation/debug controls:

- `exact_hq_diagnostics`
- `verbose`
- `orth_tol`

## Notes

- The generalized Ritz solve is now stabilized by diagonalizing the overlap
  matrix and projecting out near-singular overlap modes before solving the
  projected eigenproblem. This matters in weakly compressed threshold runs.
- Threshold mode is explicit-only; the default remains `compression_policy = "topk"`.
- Canonical current validation artifacts are:
  - `experiments/threshold_test/threshold_mode_validation_20260325`
  - `experiments/threshold_test/n2_cas10e8_threshold_sweep_20260324_v2`
- Donor-backed Python-vs-QP parity and threshold validation artifacts are
  written under `experiments/threshold_test/`.
