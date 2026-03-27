# `experiments/`

This tree is now intentionally narrow. Current sparse-policy development,
validation, and sweep artifacts live under `experiments/threshold_test/`.

## Canonical threshold artifacts

Use these directories as the current maintained references:

- `threshold_test/donor_n2_parity_sweep_20260320_rerun`
- `threshold_test/threshold_mode_validation_20260325`
- `threshold_test/qp_root_recovery_20260325`
- `threshold_test/n2_cas10e8_threshold_sweep_20260324_v2`
- `threshold_test/n2_cas12_method_compare_20260324`
- `threshold_test/n2_qp_n2_cas12_12_R2p50_threshold_sweep_20260325_pristine`
- `threshold_test/n2_qp_n2_cas14_14_R2p50_threshold_sweep_20260325_pristine`
- `threshold_test/n2_qp_ccik_simplified_smoke`
- `threshold_test/n2_qp_ccik_cas12_12_R2p50_b`
- `threshold_test/n2_qp_ccik_cas14_14_hard_budget1e7`

Everything outside `threshold_test/` was removed as part of the threshold-mode
cleanup, except for this README.

Future large-space donor-backed `N2` sweeps should use the fixed-budget driver:

- `scripts/n2_qp_fixed_budget_sweep_plots.py`
