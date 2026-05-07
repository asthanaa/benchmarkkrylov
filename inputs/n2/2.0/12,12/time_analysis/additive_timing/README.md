# N2 CAS(12,12) Additive Timing

This folder reruns the N2 CAS(12,12), R=2.00 Angstrom fixed-budget timing case
with solver-internal wall-clock instrumentation.

The additive timing report is written by:

```bash
python scripts/advanced/analyze_n2_additive_timing.py --root inputs/n2/2.0/12,12/time_analysis/additive_timing
```

The timing buckets reconcile as:

```text
total = setup + load_wf + Hq + Ritz + checkpoint + next-vector + restart + final + other
next-vector = first_orthogonalization + store/compress + second_orthogonalization + normalize + overlap_check + misc
restart = solve_Ritz + build_Ritz_vectors + store/compress + restart_orthogonalization + normalize + misc
```
