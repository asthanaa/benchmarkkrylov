# Development

## Install

For a fresh local development setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Editable install:

```bash
pip install -e .
```

With chemistry dependencies:

```bash
pip install -e ".[quantum]"
```

With tests and plotting:

```bash
pip install -e ".[quantum,plot,test]"
```

For large-space reference calculations, also create the dedicated DMRG
environment:

```bash
bash scripts/bootstrap_dmrg_env.sh
```

That environment is specified by `environment.dmrg.yml` and is created under
`.conda/cka-dmrg-py312` by default.

## Running tests

The repo now groups tests by purpose:

- `tests/beginner/`: example and public-facing smoke tests
- `tests/advanced/`: benchmark and QP script checks
- `tests/` root: core algorithm and utility tests

Run everything:

```bash
python -m pytest
```

Run only the beginner teaching checks:

```bash
python -m pytest tests/beginner
```

## Documentation site

Install docs dependencies:

```bash
pip install -e ".[docs]"
```

Serve locally:

```bash
mkdocs serve
```

Build the site:

```bash
mkdocs build
```

## Code organization rules

- beginner examples belong in `examples/`
- research workflows belong in `scripts/advanced/`
- shared advanced helpers belong in `scripts/_shared/`
- vendored Quantum Package plugin sources belong in `ccik_qp/`
- tracked teaching study definitions belong in `sample_inputs/`
- tracked Talon study inputs and shareable result files belong in `inputs/`
- raw `.ezfio/work/` runtime tensors and other large regenerable blobs stay local
- public package code belongs in `src/ccik/`

## Calculation workflow rules

- production and long calculations run on Talon, not on the local machine
- calculations should be submitted as SLURM jobs using the appropriate `talon`
  or `talon-ayush` route for that job
- keep reproducible submit scripts and input files in the repo when they are
  useful for repeating a calculation
- sync shareable Talon outputs back into the Documents checkout with:

  ```bash
  bash scripts/advanced/sync_talon_inputs_to_local.sh
  ```

- keep raw runtime blobs local; do not commit `.ezfio/work/`, oversized
  archives, or files larger than 500 MB
- after syncing safe outputs, commit in the Documents checkout and push to
  GitHub; fast-forward Talon from GitHub when Talon needs the updated tracked
  state

## Adding a new example

Add:

1. a script in `examples/scripts/`
2. a mirrored notebook in `examples/notebooks/`
3. a short entry in `docs/examples.md`
4. a smoke test in `tests/beginner/`

## Adding a new advanced workflow

Add:

1. the implementation under `scripts/advanced/`
2. a thin compatibility shim in `scripts/` only if the old path matters
3. a note in `scripts/README.md`
4. a focused advanced test if the workflow exposes stable helper functions
