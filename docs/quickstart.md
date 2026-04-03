# Quickstart

This page is the shortest path from a clean checkout to a first chemistry result.

## 1. Install

Python `3.11+` is required.

Create and activate a local virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the beginner chemistry workflows:

```bash
pip install -e ".[quantum,plot,test]"
```

## 2. Run the first example

Start with the smallest example:

```bash
python examples/scripts/01_h2_ground_state.py
```

That script:

- builds an `H2` molecule in `STO-3G`
- constructs a CAS Hamiltonian
- runs `ccik_thick`
- compares the result to exact CAS-FCI

## 3. Continue with the next examples

Compare two thick-restart budgets on a slightly larger toy system:

```bash
python examples/scripts/02_h4_compare_solvers.py
```

Then move to a realistic active-space example:

```bash
python examples/scripts/03_n2_intro_active_space.py
```

That N2 example now uses `ccik_thick`, because the teaching rule in the
notebooks is to learn the thick-restart solver first and measure its error
relative to exact CAS-FCI. The notebooks use the QP backend directly, while
the scripts remain the lightweight Python path.

## 4. Recommended solver for learners

Use `ccik_thick` unless you are intentionally comparing methods.

Why:

- it is the most stable default in this repository
- it reuses good Ritz vectors instead of restarting from scratch
- it is the solver used most often in the current examples

Even the tiny `N2` teaching notebook now follows that same rule, so new users
do not have to switch solver styles between examples.

## 5. Where the old N2 scan driver fits

The old config-driven scan entrypoint still exists:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

It is still supported, but it is no longer the recommended first step for new users.

## Next

- Vocabulary: [glossary.md](glossary.md)
- Solver guidance: [choose_solver.md](choose_solver.md)
- Full examples guide: [examples.md](examples.md)
- Config reference: [configuration.md](configuration.md)
- Large-`N2` benchmark answer: [fixed_budget_results.md](fixed_budget_results.md)
- Large-space QP setup: [reproducibility.md](reproducibility.md)
