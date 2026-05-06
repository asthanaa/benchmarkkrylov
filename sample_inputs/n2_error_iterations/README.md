# N2 Error-vs-Iteration Samples

Run from the repository root:

```bash
python sample_inputs/n2_error_iterations/run_n2_error_plots.py
```

The command runs the tracked N2 CAS(10e,10o) and CAS(12e,12o) sample inputs
with one fixed-budget setting:

- `Kprop = 1000`
- `Kstore = 50000`
- standard CCIK trace
- CKCI-0 checkpoints every 10 cumulative new Krylov vectors

The combined figure is written to:

```text
sample_inputs/n2_error_iterations/plots/n2_error_vs_iteration_k1000_s050000.png
sample_inputs/n2_error_iterations/plots/n2_error_vs_iteration_k1000_s050000.pdf
```

Each case also writes its own live trace and per-case plot next to its
`input.json`. Generated logs, plots, runtime EZFIO folders, and trace files are
ignored by git.
