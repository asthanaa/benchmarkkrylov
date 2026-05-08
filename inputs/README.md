# Study Inputs

This tree stores reproducible study definitions for the live QP iteration
drivers.

What belongs in git:

- `input.json`
- this README
- the local `.gitignore`

What does **not** belong in git:

- live plots
- iteration traces
- `system_setup/`
- runtime logs
- copied EZFIO or FCIDUMP outputs

Two bond-length groups are tracked here:

- `inputs/n2/1.1/`
  - `R = 1.10 Å`
- `inputs/n2/1.5/`
  - `R = 1.50 Å`
- `inputs/n2/2.0/`
  - `R = 2.00 Å`

For `CAS(14,14)` and larger, the tracked input definitions keep only the
`Kprop=500`, `Kstore=50000` line by default.
