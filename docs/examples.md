# Examples

The beginner examples live in two mirrored formats:

- `examples/scripts/`
- `examples/notebooks/`

The scripts are the main supported teaching path. The notebooks are there for guided exploration.

All three beginner examples are chosen so that the reported CCIK result is at
or below the chemistry rule-of-thumb target of `1 mEh` on a laptop-scale run.

## 1. H2 ground state

Files:

- `examples/scripts/01_h2_ground_state.py`
- `examples/notebooks/01_h2_ground_state.ipynb`

Concept:

- build the smallest possible CAS Hamiltonian
- run `ccik_thick`
- compare against exact CAS-FCI
- see what "much better than `1 mEh`" looks like on the simplest system

Run:

```bash
python examples/scripts/01_h2_ground_state.py
```

## 2. H4 compare solvers

Files:

- `examples/scripts/02_h4_compare_solvers.py`
- `examples/notebooks/02_h4_compare_solvers.ipynb`

Concept:

- compare `ccik`, `ccik_thick`, and `ccik_stochastic`
- see why thick restart is a useful default
- check all three errors against the `1 mEh` target line in the notebook plot

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
- use dense CCIK on this tiny active space because it reaches sub-`1 mEh`
- explain why `ccik_thick` is still the better default once the active space gets larger

Run:

```bash
python examples/scripts/03_n2_intro_active_space.py
```

## After the examples

If you want config files, sweeps, or Quantum Package workflows after finishing the examples, continue to:

- [configuration.md](configuration.md)
- [advanced.md](advanced.md)
- [fixed_budget_results.md](fixed_budget_results.md)
