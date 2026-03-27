# Configuration

Most runs use a TOML file consumed by `scripts/n2_cas_scan.py`.

Loading order:

- `--config <path>` if provided
- `configs/config.toml` if present
- packaged defaults in `src/ccik/defaults/config.toml`

## Top-level sections

- `[run]`
- `[molecule]`
- `[scan]`
- `[cas]`
- `[ccik]`
- `[ccik_thick]`
- `[ccik_stochastic]`

## `[run]`

Use either:

```toml
[run]
method = "ccik"
```

or:

```toml
[run]
methods = ["ccik", "ccik_thick", "ccik_stochastic"]
```

Valid method names:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Accepted aliases:

- `ccik_thick_restart`, `reuse_ritz`, `reuseRitz` -> `ccik_thick`
- `stochastic`, `ccikstochastic`, `fciqmckrylov`, `fciqmc_krylov` -> `ccik_stochastic`

## `[molecule]`

- `basis`: AO basis
- `unit`: geometry unit
- `charge`: total molecular charge
- `spin`: PySCF spin value `2S`
- `verbose`: PySCF verbosity

## `[scan]`

- `R_min`: minimum bond length
- `R_max`: maximum bond length
- `n_points`: number of scan points

## `[cas]`

- `ncas`: number of active orbitals
- `nelecas`: number of active electrons
- `ncore`: number of frozen doubly occupied core orbitals

## `[ccik]`

- `m`: maximum Krylov dimension
- `nadd`: number of external determinants added from the score ranking
- `nkeep`: support retained in each compressed Krylov vector
- `Kv`: stabilizer size from the largest projected amplitudes
- `selector_mode`: `cipsi` (default) or `full`
- `selector_depth`: repeated selected-CI expansion depth for `selector_mode = "cipsi"`
- `selector_ci_coeff_cutoff`: selected-CI parent threshold
- `selector_cutoff`: selected-CI determinant-generation threshold
- `orth_tol`: residual norm threshold for breakdown
- `verbose`: print iteration diagnostics

## `[ccik_thick]`

- `m_cycle`: Krylov dimension inside one restart cycle
- `ncycles`: maximum number of restart cycles
- `nroot`: number of lowest Ritz vectors reused as restart seeds
- `tol`: convergence threshold between restart energies
- `nadd`, `nkeep`, `Kv`, `selector_mode`, `selector_depth`, `selector_ci_coeff_cutoff`, `selector_cutoff`, `orth_tol`, `verbose`: same meanings as in `[ccik]`

## `[ccik_stochastic]`

- `m`, `nadd`, `nkeep`, `Kv`, `orth_tol`, `verbose`: same meanings as in `[ccik]`
- `selector_mode`: `stochastic` (default), `cipsi`, or `full`
- `selector_depth`: number of stochastic discovery / projected-contraction rounds when `selector_mode = "stochastic"`
- `n_walkers`: number of stochastic proposals per depth
- `seed`: random seed, or `null` for nondeterministic runs
- `parent_power`: parent sampling exponent
- `p_double`: probability of proposing doubles
- `mixed_double_weight`: weighting of mixed-spin doubles
- `eps_denom`: lower bound used in `|E_k - H_II|`

## Benchmarked sparse-backend settings

From `experiments/sparse_backend_benchmark_20260312/SUMMARY.md`, the following settings worked well on N2 / 6-31G / CAS(10,10) for the tested validation geometries:

### `ccik`

```toml
[ccik]
m = 32
nadd = 2000
nkeep = 12000
Kv = 4000
selector_mode = "cipsi"
selector_depth = 1
```

### `ccik_thick`

```toml
[ccik_thick]
m_cycle = 5
ncycles = 10
nroot = 2
tol = 1e-6
nadd = 4000
nkeep = 8000
Kv = 4000
selector_mode = "cipsi"
selector_depth = 1
```

### `ccik_stochastic`

```toml
[ccik_stochastic]
m = 24
nadd = 2000
nkeep = 12000
Kv = 2000
selector_mode = "stochastic"
selector_depth = 2
n_walkers = 1500
seed = 0
```

Caveat:

- These CAS(10,10) settings do not transfer cleanly to stretched N2 / CAS(12,12); the larger multireference regime still needs additional algorithmic improvement.
