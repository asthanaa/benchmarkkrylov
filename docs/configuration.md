# Configuration reference

Most runs use a TOML file consumed by `scripts/n2_cas_scan.py`.

Loading order:

- `--config <path>` if provided
- `configs/config.toml` if present
- packaged defaults in `src/ccik/defaults/config.toml`

## Top-level sections

A complete config can contain:

- `[run]`
- `[molecule]`
- `[scan]`
- `[cas]`
- `[ccik]`
- `[ccik_thick]`
- `[ccik_stochastic]`

## `[run]`

Controls which solver or solvers are executed.

Supported shapes:

```toml
[run]
method = "ccik"
```

```toml
[run]
methods = ["ccik", "ccik_thick", "ccik_stochastic"]
```

Valid method names:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Aliases accepted by the driver:

- `ccik_thick_restart`, `reuse_ritz`, `reuseRitz` -> `ccik_thick`
- `stochastic`, `ccikstochastic`, `fciqmckrylov`, `fciqmc_krylov` -> `ccik_stochastic`

## `[molecule]`

Controls the PySCF molecule used to build the CAS Hamiltonian.

- `basis`: AO basis label
- `unit`: geometry unit, usually `Angstrom`
- `charge`: total molecular charge
- `spin`: PySCF spin value `2S`
- `verbose`: PySCF verbosity level

The N2 geometry itself is generated in the driver from the scan bond length.

## `[scan]`

Controls the bond scan grid.

- `R_min`: minimum bond length in Angstrom
- `R_max`: maximum bond length in Angstrom
- `n_points`: number of scan points, including both endpoints

The driver uses `numpy.linspace(R_min, R_max, n_points)`.

## `[cas]`

Defines the active space used to build the CAS Hamiltonian.

- `ncas`: number of active orbitals
- `nelecas`: number of active electrons
- `ncore`: number of frozen doubly occupied core orbitals

For singlets, the example drivers enforce `mol.nelectron == 2*ncore + nelecas`.

## `[ccik]`

Parameters for baseline sparse CCIK.

- `m`: maximum Krylov dimension
- `nadd`: number of external determinants added from the score ranking at each step
- `nkeep`: maximum support retained in each compressed Krylov vector
- `Kv`: stabilizer size; retain the top-`Kv` projected amplitudes
- `selector_mode`: candidate-discovery / projected-matvec mode: `cipsi` (default) or `full`
- `selector_depth`: number of repeated selected-CI expansion layers used to build each projected `H|q_k>`
- `selector_ci_coeff_cutoff`: parent coefficient threshold used by the PySCF selected-CI expander
- `selector_cutoff`: determinant generation threshold used by the PySCF selected-CI expander
- `orth_tol`: residual norm threshold used to detect Krylov breakdown
- `verbose`: print per-iteration diagnostics

## `[ccik_thick]`

Parameters for CCIK with thick restart.

- `m_cycle`: Krylov dimension built inside one restart cycle
- `ncycles`: maximum number of restart cycles
- `nroot`: number of lowest Ritz vectors reused as restart seeds
- `tol`: convergence threshold on successive cycle energies
- `nadd`, `nkeep`, `Kv`, `selector_mode`, `selector_depth`, `selector_ci_coeff_cutoff`, `selector_cutoff`, `orth_tol`, `verbose`: same meanings as in `[ccik]`

## `[ccik_stochastic]`

Parameters for CCIK-stochastic.

- `m`, `nadd`, `nkeep`, `Kv`, `selector_ci_coeff_cutoff`, `selector_cutoff`, `orth_tol`, `verbose`: same meanings as in `[ccik]`
- `selector_mode`: `stochastic` (default), `cipsi`, or `full`
- `selector_depth`: repeated discovery / projected-contraction depth when `selector_mode = "stochastic"`
- `n_walkers`: number of stochastic proposals used at each depth
- `seed`: random seed for reproducibility; use `null` for nondeterministic runs
- `parent_power`: parent sampling exponent, where parents are weighted by `|q_J|**parent_power`
- `p_double`: probability of proposing a double excitation instead of a single excitation
- `mixed_double_weight`: relative weight of mixed-spin doubles among all doubles
- `eps_denom`: lower bound used in the score denominator `|E_k - H_II|`

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
