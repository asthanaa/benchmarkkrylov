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

1. Create a local Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. Install the package and PySCF:

   ```bash
   pip install -e ".[quantum,plot,test]"
   ```

3. Run the first example:

   ```bash
   python examples/scripts/01_h2_ground_state.py
   ```

4. Then continue with:

- [Beginner docs index](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Chemistry glossary](docs/glossary.md)
- [Examples guide](docs/examples.md)

If you only want the beginner Python examples, you can stop there.

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
- [Reproducible setup](docs/reproducibility.md)
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
- `ccik_qp/`: vendored Quantum Package plugin source
- `inputs/`: tracked study definitions for live QP runs
- `tests/`: beginner, core, and advanced checks

## Documentation site

The MkDocs source lives in `docs/`. The main starting pages are:

- `docs/index.md`
- `docs/quickstart.md`
- `docs/glossary.md`
- `docs/examples.md`
- `docs/reproducibility.md`

## Reproducible large-space setup

For the maintained large-space `N2` workflow on a fresh machine:

1. clone the repo and enter it:

   ```bash
   git clone https://github.com/asthanaa/CompressK.git
   cd CompressK
   ```

2. create and activate the main Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. install the repo package and local tooling:

   ```bash
   pip install -e ".[quantum,plot,test]"
   ```

4. create the dedicated DMRG environment:

   ```bash
   bash scripts/bootstrap_dmrg_env.sh
   ```

5. source Quantum Package on your machine:

   ```bash
   source /path/to/quantum_package.rc
   ```

6. rebuild the vendored QP plugin into your Quantum Package installation:

   ```bash
   python scripts/advanced/qp_ccik_bridge.py rebuild-plugin
   ```

7. launch a representative QP `N2` study:

   ```bash
   python scripts/advanced/n2_thick_iteration_study.py \
     --input inputs/n2/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3/input.json
   ```

8. open the generated plots:

   ```bash
   CASE=inputs/n2/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3
   open "$CASE/energy_error_vs_iteration_comparison.png"
   open "$CASE/k0500_s050000/energy_error_vs_iteration.png"
   open "$CASE/k1000_s100000/energy_error_vs_iteration.png"
   ```

The DMRG bootstrap spec is tracked in `environment.dmrg.yml`. The binary
environment under `.conda/` is local-only and intentionally not committed.
