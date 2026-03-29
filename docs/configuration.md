# Configuration Reference

You only need this page for config-driven workflows.

If you are just learning the code, start with the example scripts instead of the config system.

## Where config files are used

The legacy-compatible scan driver still accepts TOML:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

Loading order:

- `--config <path>` if provided
- `configs/config.toml` if present
- packaged defaults in `src/ccik/defaults/config.toml`

## Top-level sections

- `[run]`
- `[molecule]`
- `[scan]`
- `[cas]`
- `[ccik]`
- `[ccik_thick]`
- `[ccik_stochastic]`

## `[run]`

Choose one method:

```toml
[run]
method = "ccik_thick"
```

Or compare several:

```toml
[run]
methods = ["ccik", "ccik_thick", "ccik_stochastic"]
```

The recommended default for most users is `ccik_thick`.

## `[molecule]`

Controls the PySCF molecule:

- `basis`
- `unit`
- `charge`
- `spin`
- `verbose`

## `[scan]`

Controls the bond-length grid for scan-style drivers:

- `R_min`
- `R_max`
- `n_points`

## `[cas]`

Defines the active space:

- `ncas`
- `nelecas`
- `ncore`

Example:

```toml
[cas]
ncas = 10
nelecas = 10
ncore = 2
```

## `[ccik]`, `[ccik_thick]`, `[ccik_stochastic]`

These sections store solver-specific parameters.

For modern sparse runs, the most important learner-facing knobs are:

- `compression_policy`
- `fixed_store_keep`
- `fixed_propagate_keep`

The repository-wide learner-friendly defaults are:

- `compression_policy = "fixed_budget"`
- `fixed_store_keep = 65536`
- `fixed_propagate_keep = 512`

For the current large-`N2` benchmark strategy, keep `fixed_store_keep = 65536`
and tune `fixed_propagate_keep` first. The detailed results are summarized in
[fixed_budget_results.md](fixed_budget_results.md).
