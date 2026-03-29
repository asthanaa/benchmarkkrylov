# Advanced Scripts

This directory is now the **advanced workflow surface**.

If you are learning the project for the first time, start in:

- `examples/`
- `docs/quickstart.md`

## Layout

- `scripts/advanced/`: benchmark, sweep, parity, and Quantum Package drivers
- `scripts/_shared/`: helpers reused by those advanced drivers
- `scripts/*.py`: thin compatibility shims for older command names

## Typical advanced tasks

### Quantum Package bridge and plugin tooling

- `scripts/advanced/qp_ccik_bridge.py`
- `scripts/_shared/qp_ccik_common.py`

### N2 benchmark and sweep workflows

- `scripts/advanced/n2_qp_fixed_budget_sweep_plots.py`
- `scripts/advanced/n2_qp_fixed_budget_benchmark_summary.py`
- `scripts/advanced/n2_qp_threshold_sweep_plots.py`
- `scripts/advanced/n2_accuracy_gate.py`

### Cr2 workflows

- `scripts/advanced/cr2_qp_fixed_budget_sweep_plots.py`
- `scripts/advanced/cr2_qp_ccik_tune.py`
- `scripts/advanced/cr2_qp_ccik_refine.py`

### Validation and parity

- `scripts/_shared/debug_selector_parity.py`
- `scripts/advanced/threshold_mode_validation.py`
- `scripts/advanced/validate_selector_modes.py`

## Compatibility note

Old command paths such as:

```bash
python scripts/n2_qp_fixed_budget_sweep_plots.py
```

still work through shim files, but the implementation now lives in `scripts/advanced/`.
