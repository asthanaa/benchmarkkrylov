# Examples

The beginner examples live in two mirrored formats:

- `examples/scripts/`
- `examples/notebooks/`

The scripts are the plain-Python path. The notebooks are the guided teaching
path, and they now use the Quantum Package backend directly.

Important cleanup note:

- the notebooks now contain the actual teaching code in their own code cells
- they do not hide the work inside an imported script file
- the script files still exist as fast command-line companions
- the notebook path is now the QP-backed production-style teaching path

The three beginner notebooks are chosen to teach the QP workflow clearly on
laptop-scale runs. The `N2` notebook is tuned to land well below `1 mEh`,
while the `H2` parity notebook and the `H4` budget notebook are mainly there
to explain backend agreement and budget trends.

## 1. H2 ground state

Files:

- `examples/scripts/01_h2_ground_state.py`
- `examples/notebooks/01_h2_ground_state.ipynb`

Concept:

- build the smallest possible CAS Hamiltonian
- run `ccik_thick` through the QP backend
- compare against exact CAS-FCI
- check that the QP and Python thick-restart implementations agree to machine precision on the same FCIDUMP
- read the shortest Python version in the script and the fuller QP workflow in the notebook

Run:

```bash
python examples/scripts/01_h2_ground_state.py
```

## 2. H4 thick-restart comparison

Files:

- `examples/scripts/02_h4_compare_solvers.py`
- `examples/notebooks/02_h4_compare_solvers.ipynb`

Concept:

- compare a lighter and a refined QP `ccik_thick` fixed-budget setting
- see why thick restart is the useful default to learn first
- check both errors against exact CAS-FCI in the notebook plot
- learn how larger `Kstore` and `Kprop` budgets usually lower the error

Run:

```bash
python examples/scripts/02_h4_compare_solvers.py
```

## 3. N2 active-space introduction

Files:

- `examples/scripts/03_n2_intro_active_space.py`
- `examples/notebooks/03_n2_intro_active_space.ipynb`

Concept:

- introduce a realistic molecule
- explain what `CAS(6e,6o)` means in practice
- use `ccik_thick` even on this tiny active space so the teaching path matches
  the production default
- make `Kstore` and `Kprop` explicit instead of hiding them behind top-k knobs
- measure the solver error directly against exact CAS-FCI
- keep the fully QP-backed notebook small and readable by using a minimal-basis `N2` teaching case

Run:

```bash
python examples/scripts/03_n2_intro_active_space.py
```

## After the examples

If you want config files, sweeps, or Quantum Package workflows after finishing the examples, continue to:

- [configuration.md](configuration.md)
- [advanced.md](advanced.md)
- [fixed_budget_results.md](fixed_budget_results.md)
