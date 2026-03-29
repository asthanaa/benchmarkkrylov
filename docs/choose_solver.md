# Choose a Solver

This repository exposes three solver families.

## Recommended default: `ccik_thick`

Choose `ccik_thick` if:

- you are new to the code
- you want the most robust general-purpose option in this repo
- you want the examples to look like the recommended workflow
- you want the same solver branch used in the validated large-`N2` QP benchmarks

Why it is the default teaching choice:

- it keeps several useful Ritz vectors during restart
- it is more forgiving than the plain baseline solver
- it is the best match to the current fixed-budget defaults

## `ccik`

Choose `ccik` if:

- you want the simplest deterministic baseline
- you want to study the core Krylov iteration without restart logic

This is the easiest solver to explain mathematically, but it is not the most stable production choice for learners.

For very small teaching systems, it can still be the right notebook choice
because it may already reach exact or near-exact CAS-FCI energies without any
restart machinery. The `N2 CAS(6e,6o)` notebook uses it for that reason.

## `ccik_stochastic`

Choose `ccik_stochastic` if:

- you want to study stochastic candidate discovery
- you care about that specific algorithmic variant

This is more advanced and is not the first solver to learn.

## Compression policy

For modern sparse runs in this repository, the recommended compression policy is:

- `fixed_budget`

The repository-wide learner-friendly defaults are:

- `fixed_store_keep = 65536`
- `fixed_propagate_keep = 512`

These are teaching defaults, not universal chemistry truths. Larger active spaces may require larger budgets.

## Current production strategy for large N2 benchmarks

The finished hard-point `N2` studies in this repo point to the following
production choice:

- QP backend
- `matvec_backend = "connected_sd_fullq"`
- `selector_mode = "abs_residual"`
- `method = "ccik_thick"`
- `compression_policy = "fixed_budget"`

For the compression pair itself, the current transfer strategy is:

1. set `fixed_store_keep = 65536`
2. scan `fixed_propagate_keep = 512, 1024, 2048`
3. only then try a larger store budget such as `131072`

The reason is empirical: on the completed `N2 CAS(12e,12o)` and
`CAS(14e,14o)` runs, the propagate budget is the sharper tuning axis while the
store budget is already good once it reaches `65536`.

For the actual finished numbers, see [fixed_budget_results.md](fixed_budget_results.md).
