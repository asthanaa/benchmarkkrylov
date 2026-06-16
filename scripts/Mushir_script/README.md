# Mushir QRTE/QKUD 0.2 Plot Workflow

This folder contains the Talon submission workflow for a second set of
MolVQE-21 QRTE/QKUD plots at:

- QRTE `delta_t = 0.2`
- QKUD `epsilon = 0.2`

The workflow writes separate result directories and plot files so the original
`delta_t = 0.1` and `epsilon = 0.1` plots are not overwritten.

## Output Locations

Run outputs:

```text
runs/qrte_natorb_dt0.2
runs/qkud_natorb_eps0.2
```

Plot outputs:

```text
plots/mushir_dt0p2_eps0p2/adapt_vs_qrte_dt0p2_iterations_vs_entropy.csv
plots/mushir_dt0p2_eps0p2/adapt_vs_qrte_dt0p2_iterations_vs_entropy.png
plots/mushir_dt0p2_eps0p2/adapt_vs_qrte_dt0p2_iterations_vs_entropy.pdf
plots/mushir_dt0p2_eps0p2/adapt_vs_qkud_eps0p2_iterations_vs_entropy.csv
plots/mushir_dt0p2_eps0p2/adapt_vs_qkud_eps0p2_iterations_vs_entropy.png
plots/mushir_dt0p2_eps0p2/adapt_vs_qkud_eps0p2_iterations_vs_entropy.pdf
```

The CSV files use the same columns as the main plot tables. Cases with no
completed summary yet appear as `method_status=missing`; completed cases update
one at a time as the live watcher sees new `summary.json` files.

## Submit On Talon

From the repository root:

```bash
scripts/Mushir_script/submit_talon_dt0p2_eps0p2.sh
```

The submitter starts three jobs:

- QRTE pool job for `runs/qrte_natorb_dt0.2`
- QKUD pool job for `runs/qkud_natorb_eps0.2`
- live plot watcher for `plots/mushir_dt0p2_eps0p2`

The watcher starts immediately and refreshes whichever method has new completed
summaries. It exits after both watched Slurm jobs leave the queue and one final
refresh has run.

Useful overrides:

```bash
QCANT_N_STEPS=200 scripts/Mushir_script/submit_talon_dt0p2_eps0p2.sh
QCANT_CONCURRENCY=12 QCANT_MAX_WORKERS=3 scripts/Mushir_script/submit_talon_dt0p2_eps0p2.sh
```

## Manual Plot Refresh

Refresh both plots once:

```bash
python scripts/Mushir_script/collect_and_plot_dt0p2_eps0p2.py --method both
```

Watch for new summaries without submitting jobs:

```bash
python scripts/Mushir_script/watch_dt0p2_eps0p2_plots.py \
  --method both \
  --interval-seconds 60 \
  --allow-empty-refresh
```
