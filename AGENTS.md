# Agent Workflow Rules

## Working copy

- Use `/Users/ayushasthana/Documents/CompressK-edit` as the local working repo.
- Do not update OneDrive copies unless explicitly asked.

## Calculation workflow

- Run production or long calculations on Talon, not on the local machine.
- Submit calculations as SLURM jobs on Talon using the appropriate `talon` or `talon-ayush` route for the job.
- Keep SLURM submit scripts and reproducible input files in the repo when they are useful for repeating a calculation.
- After a Talon run finishes, sync shareable outputs back into the Documents checkout, respecting the 500 MB per-file limit.
- Keep raw runtime blobs local. Do not commit `.ezfio/work/`, oversized archives, or any file larger than 500 MB.
- Commit safe result files in the Documents checkout, push them to GitHub, then fast-forward Talon from GitHub when Talon needs the updated tracked state.
