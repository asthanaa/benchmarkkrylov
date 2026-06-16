# Dataset And Manifest

The benchmark database comes from Zenodo record `19051536`. The repository keeps
both the raw archive and the extracted JSON tree so the manifest and run inputs
can be reproduced.

## Database Layout

The database lives under:

```text
dataset/zenodo_19051536
```

Important paths:

- `raw/data_repository.7z`: original archived database.
- `raw/record_metadata.json`: metadata for the record.
- `extracted/data_repository`: extracted database used by scripts.
- `extracted/data_repository/MolVQE-21`: top-level MolVQE-21 JSON systems used by the canonical benchmark.
- `extracted/data_repository/6_Orbs(Fig._3,11,13)`: smaller six-orbital cases.
- `extracted/data_repository/VQE_comparison(Fig.7)`: VQE comparison data.
- `extracted/data_repository/stretched_H_chains(Fig._6,12)`: stretched hydrogen-chain cases.
- `extracted/data_repository/data_format.txt`: source format notes.

## Canonical Manifest

The benchmark drivers read:

```text
manifests/molvqe21_fig1b_manifest.csv
```

This CSV is built from the JSON files directly under
`dataset/zenodo_19051536/extracted/data_repository/MolVQE-21`.

The manifest builder is:

```bash
python scripts/build_manifest.py
```

By default it writes `manifests/molvqe21_fig1b_manifest.csv`. It can also accept
custom paths:

```bash
python scripts/build_manifest.py \
  --data-root dataset/zenodo_19051536/extracted/data_repository \
  --output manifests/molvqe21_fig1b_manifest.csv
```

## Manifest Fields

Each row is one benchmark case. Important fields are:

- `case_id`: filesystem-safe case name used under `runs/`.
- `source_name`: original JSON filename.
- `source_json_path`: path back to the source database JSON.
- `symbols`: compact JSON list of atomic symbols.
- `geometry_bohr`: compact JSON geometry in Bohr.
- `geometry_angstrom`: compact JSON geometry in Angstrom.
- `basis`: basis set from the source JSON.
- `charge` and `spin`: molecular charge and spin.
- `active_electrons` and `active_orbitals`: active-space size.
- `casscf_energy_hartree`: reference CASSCF energy.
- `renyi_alpha`: entropy parameter; currently `0.25`.
- `h0.25`: Renyi entropy computed from the CASSCF CI vector.
- `chemical_accuracy_hartree`: chemical-accuracy threshold, `0.0016 Eh`.
- `adapt_chem_accuracy_iteration`: first stored ADAPT iteration reaching chemical accuracy.
- `has_adapt_chem_hit`: whether ADAPT reaches chemical accuracy in the source spans.
- `final_num_iterations` and `final_reason`: final status from the source JSON.

## How Geometry And Entropy Are Derived

`scripts/build_manifest.py` reads each source JSON, extracts molecular metadata,
converts coordinates from Bohr to Angstrom with `0.529177210903`, and computes
`h0.25` from the squared CASSCF CI coefficients. The same script records the
first ADAPT iteration whose energy is within `0.0016 Eh` of the CASSCF energy.

The runners use the manifest rather than re-parsing the source JSON during every
case. Each run still writes `input_snapshot.json` so the exact manifest row used
for that run is preserved with the output.
