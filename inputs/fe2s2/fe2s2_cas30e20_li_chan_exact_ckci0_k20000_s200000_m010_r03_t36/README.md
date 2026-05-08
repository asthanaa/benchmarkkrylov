# [2Fe-2S] CAS(30e,20o) CK/CKCI-0 Setup

This folder is configured for a 36-core Talon-large Slurm run.

## Hamiltonian And Reference

The FCIDUMP was copied from the paper-linked repository:

- Paper: https://arxiv.org/html/2501.07231v4
- Hamiltonian repository: https://github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters
- Repository file: `Fe2S2_and_Fe4S4/Fe2S2/fe2s2`

Reference values used in `input.json`:

- Active-space exact reference: `E_act = -116.6056091 Eh`
- Core energy: `E_core = -4976.26532397 Eh`
- Total reference: `E_total = -5092.87093307 Eh`

The downloaded FCIDUMP header is `NORB=20`, `NELEC=30`, `MS2=0`. Its embedded `ECORE` is zero, so the runner explicitly adds the paper core energy when reporting total energies and errors.

## Recommended Run

The prepared run uses:

- `Kprop = 20000`
- `Kstore = 200000`
- `m_cycle = 10`
- `nroot = 3`
- `ncycles = 20`
- `max_cumulative_iterations = 100`
- `exactized_checkpoint_every = 1`
- `omp_threads = 36`

This is intentionally CK plus CKCI-0 every Krylov iteration. CKCI-0 here means diagonalizing the Hamiltonian in the union of unique determinants accumulated across the current Krylov basis at each checkpoint; it is diagnostic and does not feed back into the CK restart vectors.

## Why These Settings

From the recent N2, Cr2, and polyene runs, the practical pattern is:

- `m_cycle=31` is good for plain CK traces, but too heavy when exactizing at every iteration.
- `m_cycle=10` with `nroot=3` is the safer cadence for CKCI-0 style diagnostics on expensive systems.
- Cr2 CKCI-0 was set at `Kprop=20000`, `Kstore=50000`; Fe2S2 CAS(30e,20o) is a larger active-electron problem, so this setup keeps the same broad propagation budget and increases the retained store to `200000`.
- If this is too slow, the first fallback should be `Kprop=10000`, `Kstore=50000` with the same `m_cycle=10`, `nroot=3`.

## Before Launch

Validate without running chemistry:

```bash
python run_fe2s2_python_fcidump_ckci0.py --input input.json --check-only
```

Submit on Talon:

```bash
sbatch submit_fe2s2_ckci0_talon_large.sbatch
```

Expected live outputs after launch:

- `ccik_stdout.log`
- `ccik_stderr.log`
- `iteration_trace.csv`
- `iteration_trace.json`
- `live_status.json`
- `summary.json`
- `study_summary.json`
- `energy_error_vs_iteration.png`
- `energy_error_vs_iteration.pdf`
- `solver_stats.json`
