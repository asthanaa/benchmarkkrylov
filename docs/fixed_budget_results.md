# Fixed-Budget N2 Results

This page records the current production answer for the large-active-space
`N2` studies in this repository.

## Ground truth

For the validated `N2` hard-point benchmarks, the best current strategy is:

- Quantum Package backend
- `matvec_backend = "connected_sd_fullq"`
- `selector_mode = "abs_residual"`
- `method = "ccik_thick"`
- `compression_policy = "fixed_budget"`

The open tuning question is therefore not the selector. It is the fixed-budget
compression pair:

- `fixed_store_keep`
- `fixed_propagate_keep`

## Main conclusion

The transfer lesson from `CAS(12e,12o)` to `CAS(14e,14o)` is simple:

1. keep `fixed_store_keep` moderate
2. tune `fixed_propagate_keep` first
3. only increase `fixed_store_keep` if the propagate scan plateaus

On both finished `N2` benchmarks, `fixed_store_keep = 65536` is already in the
right regime. The propagate budget is the more important knob.

## Benchmark summary

### N2 / 6-31G / R = 2.50 Å / CAS(12e,12o)

Reference used in the finished sweep:

- `reference_kind = dmrg`
- `reference_label = DMRG`

Best completed thick points:

| Thick config | `fixed_store_keep` | `fixed_propagate_keep` | `|ΔE|` vs reference | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `restart_30_30_8` | 65536 | 2048 | `0.000779 mEh` | `3607.41 s` |
| `restart_30_30_8` | 65536 | 1024 | `0.010761 mEh` | `1841.92 s` |
| `restart_30_30_8` | 65536 | 256 | `0.011361 mEh` | `1109.53 s` |

Practical readout:

- `2048` is the best accuracy point
- `1024` is almost as good and materially faster
- `256` is still excellent on this system

### N2 / 6-31G / R = 2.50 Å / CAS(14e,14o)

Reference used in the finished sweep:

- `reference_kind = exact_fci`
- `reference_label = Exact FCI`

Best completed thick points:

| Thick config | `fixed_store_keep` | `fixed_propagate_keep` | `|ΔE|` vs reference | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `restart_16_16_5` | 65536 | 1024 | `0.712248 mEh` | `2771.71 s` |
| `restart_16_16_5` | 131072 | 2048 | `0.714092 mEh` | `7181.00 s` |
| `restart_16_16_5` | 65536 | 2048 | `0.717310 mEh` | `3818.81 s` |
| `restart_16_16_5` | 65536 | 512 | `0.732518 mEh` | `1738.92 s` |

Practical readout:

- the same `fixed_store_keep = 65536` used in `CAS(12e,12o)` still works well
- the best completed point shifted to `fixed_propagate_keep = 1024`
- increasing store beyond `65536` did not buy a clear improvement

## Strategy to reuse on larger N2 active spaces

If you want a robust first pass on the same `N2` family:

1. use `ccik_thick`
2. use the QP `connected_sd_fullq` backend
3. keep `selector_mode = "abs_residual"`
4. start with `compression_policy = "fixed_budget"`
5. set `fixed_store_keep = 65536`
6. scan `fixed_propagate_keep = 512, 1024, 2048`
7. only then try `fixed_store_keep = 131072`

This is the strategy supported by the completed benchmarks on disk.

## Consolidated benchmark plot

The figure below combines the finished store and propagate sweeps for the two
large `N2` cases.

![Consolidated N2 fixed-budget strategy](assets/figures/n2_fixed_budget_strategy_summary.png)

The pattern is consistent across both systems:

- the store scan is relatively flat once `fixed_store_keep` reaches `65536`
- the propagate scan is the sharper tuning axis
- thick restart remains the right solver branch

## Original sweep plots

### CAS(12e,12o)

![CAS(12e,12o) store sweep](assets/figures/n2_cas12_fixed_store_keep_sweep.png)

![CAS(12e,12o) propagate sweep](assets/figures/n2_cas12_fixed_propagate_keep_sweep.png)

### CAS(14e,14o)

![CAS(14e,14o) store sweep](assets/figures/n2_cas14_fixed_store_keep_sweep.png)

![CAS(14e,14o) propagate sweep](assets/figures/n2_cas14_fixed_propagate_keep_sweep.png)

## Source artifacts

Raw benchmark tables:

- `experiments/threshold_test/n2_qp_n2_cas12_12_R2p50_fixed_budget_centered_20260326_live/fixed_budget_sweep.csv`
- `experiments/threshold_test/n2_qp_n2_cas14_14_R2p50_fixed_budget_centered_20260326_thick_prop2048_live/fixed_budget_sweep.csv`

Figure generator:

- `scripts/advanced/n2_qp_fixed_budget_benchmark_summary.py`
