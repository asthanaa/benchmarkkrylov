# CompressK

`CompressK` is a teaching-and-research repository for the **Compressed Krylov in Configuration Interaction (CCIK)** family of solvers.

The repository now has two clearly labeled paths:

- **Start Here: Chemistry Student**
  - Learn the basic ideas with small molecules.
  - Run short examples that compare CCIK to exact CAS-FCI.
  - Use `ccik_thick` as the recommended default solver.
- **Advanced / Research Workflows**
  - Large-system benchmarks
  - Quantum Package bridge tooling
  - Fixed-budget and threshold sweeps
  - Parity and validation scripts

## Start Here: Chemistry Student

1. Install the package and PySCF:

   ```bash
   pip install -e ".[quantum]"
   ```

2. Run the first example:

   ```bash
   python examples/scripts/01_h2_ground_state.py
   ```

3. Then continue with:

- [Beginner docs index](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Chemistry glossary](docs/glossary.md)
- [Examples guide](docs/examples.md)

## Recommended learning path

- `H2`: build the smallest CAS Hamiltonian and compare CCIK to exact CAS-FCI
- `H4`: compare lighter and refined fixed-budget thick settings
- `N2`: learn how active-space choices enter a realistic chemistry example

The mirrored teaching files live in:

- `examples/scripts/`
- `examples/notebooks/`

The notebooks are written as self-contained teaching notebooks. They now show
the actual example code inline instead of importing a hidden script, and they
teach the Quantum Package backend directly. The companion scripts stay as the
fast plain-Python path.

## Advanced / Research Workflows

Advanced scripts now live under:

- `scripts/advanced/`
- `scripts/_shared/`

For convenience, the old root-level script names still exist as compatibility shims. New documentation points to the cleaner advanced layout instead of treating those workflows as the default starting point.

Useful advanced references:

- [Advanced workflows](docs/advanced.md)
- [Fixed-budget N2 benchmark results](docs/fixed_budget_results.md)
- [Quantum Package backend](docs/quantum_package_backend.md)
- [Advanced scripts index](scripts/README.md)
- [Experiment artifact notes](experiments/README.md)

## Current production answer

The current large-active-space benchmark conclusion is:

- use the Quantum Package path
- use `connected_sd_fullq`
- use `abs_residual`
- use `ccik_thick`
- tune `fixed_store_keep` and `fixed_propagate_keep`

The finished `N2` hard-point studies show that the compression problem is still
mainly a budget-tuning problem. The repository now keeps exactly two selector
datasets:

- the production `abs_residual` baseline
- the single-pass `pt2_residual` comparison run

The current best strategy is to keep `fixed_store_keep` moderate, start at
`65536`, and tune `fixed_propagate_keep` first.

## Package overview

The public Python API stays small:

- `ccik_ground_energy_dense(...)`
- `ccik_ground_energy_dense_thick_restart(...)`
- `ccik_ground_energy_stochastic(...)`
- `make_mol_pyscf(...)`
- `build_cas_hamiltonian_pyscf(...)`
- `exact_cas_fci_energy_pyscf(...)`

For most learners, `ccik_ground_energy_dense_thick_restart(...)` is the best place to start.

## Repository layout

- `src/ccik/`: importable package
- `examples/`: beginner scripts and notebooks
- `docs/`: narrative documentation
- `scripts/advanced/`: research, benchmark, and QP workflows
- `scripts/_shared/`: helpers used by advanced scripts
- `tests/`: beginner, core, and advanced checks

## Documentation site

The MkDocs source lives in `docs/`. The main starting pages are:

- `docs/index.md`
- `docs/quickstart.md`
- `docs/glossary.md`
- `docs/examples.md`
