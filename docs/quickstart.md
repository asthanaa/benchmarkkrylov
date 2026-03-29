# Quickstart

This page is the shortest path from a clean checkout to a first chemistry result.

## 1. Install

Python `3.11+` is required.

For the beginner chemistry workflows:

```bash
pip install -e ".[quantum]"
```

If you also want plotting and tests:

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

Compare solver styles on a slightly larger toy system:

```bash
python examples/scripts/02_h4_compare_solvers.py
```

Then move to a realistic active-space example:

```bash
python examples/scripts/03_n2_intro_active_space.py
```

That N2 example deliberately uses the plain dense solver, not `ccik_thick`,
because the active space is still small enough that dense CCIK reaches
sub-`1 mEh` agreement with exact CAS-FCI.

## 4. Recommended solver for learners

Use `ccik_thick` unless you are intentionally comparing methods.

Why:

- it is the most stable default in this repository
- it reuses good Ritz vectors instead of restarting from scratch
- it is the solver used most often in the current examples

One exception:

- for the tiny `N2 CAS(6e,6o)` teaching notebook, dense CCIK is used because it
  already reaches the `1 mEh` chemistry target and keeps the explanation simple

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
