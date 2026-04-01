# Examples

This directory is the recommended starting point for new users.

## Layout

- `examples/scripts/`: plain Python scripts that can be run from the command line
- `examples/notebooks/`: self-contained QP-backed notebooks for guided exploration

The two formats are intentionally similar, but they serve different purposes:

- the script is the shortest runnable version
- the notebook is the teaching version with inline code and explanation
- the script stays on the fast Python backend, while the notebook shows the QP workflow

## Suggested order

1. `01_h2_ground_state`
2. `02_h4_compare_solvers`
3. `03_n2_intro_active_space`

All three are meant to be easy to read. The N2 notebook is tuned to be well
below `1 mEh`, while the H2 and H4 notebooks focus on backend parity and
budget trends.

For the current large-`N2` selector benchmark story, see:

- `docs/fixed_budget_results.md`
