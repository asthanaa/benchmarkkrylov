# Results And Provenance

This repository keeps generated benchmark results so the plotted comparisons can
be inspected without rerunning every case.

## Run Directories

`runs/` stores case-level outputs. Typical production directories include:

- `runs/qrte_dt0.1`: QRTE runs at `delta_t=0.1`.
- `runs/qkud_eps0.1`: QKUD runs at `epsilon=0.1`.
- `runs/qrte_natorb_dt0.1`: QRTE natural-orbital runs.
- `runs/qkud_natorb_eps0.1`: QKUD natural-orbital runs.
- `runs/qrte_natorb_delta_sweep`: QRTE delta sweep results.
- `runs/qkud_natorb_epsilon_sweep`: QKUD epsilon sweep results.

Within a case directory, the important files are:

- `input_snapshot.json`: manifest row and molecule data used by the run.
- `run_config.json`: method, algorithm parameters, cache location, and runtime settings.
- `min_energy_history.csv`: per-iteration minimum projected energy.
- `summary.json`: status, hit iteration, final energy, elapsed time, and cache metadata.
- `stdout.log` and `stderr.log`: child process logs for direct driver runs.
- `pool_launch.json`, `pool_stdout.log`, and `pool_stderr.log`: pool-run metadata and logs.

## Summary Semantics

Each completed `summary.json` includes:

- `status`: usually `completed` or `failed`.
- `method`: `qrte` or `qkud`.
- `hit_iteration`: first iteration within chemical accuracy, or null if not reached.
- `hit_chemical_accuracy`: boolean hit flag.
- `n_steps_requested` and `n_steps_completed`: requested and actual basis-growth steps.
- `stopped_early`: whether early stopping ended the run after a hit.
- `final_projected_ground_energy_hartree`: final projected ground-state estimate.
- `final_min_history_energy_hartree`: final value in the minimum-energy history.
- `hamiltonian_source` and `hamiltonian_cache_dir`: Hamiltonian construction and cache information.

Chemical accuracy is `0.0016 Eh` relative to the CASSCF energy from the manifest.

## Plots

`plots/` contains comparison tables and figures:

- `adapt_vs_qrte_iterations_vs_entropy.csv`
- `adapt_vs_qrte_iterations_vs_entropy.png`
- `adapt_vs_qrte_iterations_vs_entropy.pdf`
- `adapt_vs_qrte_iterations_vs_entropy_sweep_points.csv`
- `adapt_vs_qkud_iterations_vs_entropy.csv`
- `adapt_vs_qkud_iterations_vs_entropy.png`
- `adapt_vs_qkud_iterations_vs_entropy.pdf`
- `adapt_vs_qkud_iterations_vs_entropy_sweep_points.csv`
- `live_plot_watch_status.json`

The plot CSV files combine manifest data, ADAPT reference iterations, QRTE or
QKUD hit iterations, and sweep summaries. The PNG/PDF files visualize iteration
counts against `h0.25` entropy.

## Logs

`logs/` stores Slurm stdout and stderr files. These are useful when a
`summary.json` is missing or marked failed, because they capture scheduler-level
and process-level failures outside the per-case output directory.

## Hamiltonian Cache

`hamiltonian_cache/casscf_natorb` stores cached Hamiltonian arrays, primarily
`.npz` files. Reusing this cache avoids rebuilding expensive active-space
Hamiltonians across reruns and parameter sweeps.

## Provenance

`provenance/` records how the workspace evolved during long benchmark runs. It
contains restart notes, source status snapshots, patched-file copies, checksum
records, and job-id markers. Treat these files as the audit trail for why a
particular restart or recovery run exists.
