# `configs/`

This folder is for optional user override configs.

The package ships with a default config at:

- `src/ccik/defaults/config.toml`

Config files are mainly useful for the legacy-compatible scan driver and other
scripted workflows. New learners should start with the files in `examples/`
instead of writing TOML immediately.

To override defaults, create your own TOML file here and run:

```bash
python scripts/n2_cas_scan.py --config configs/my_config.toml
```
