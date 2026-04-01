# Fixed-Budget N2 Selector Results

This page records the two `N2` benchmark datasets that the repository now keeps
as the local benchmark ground truth:

- `abs_residual`: the validated production selector
- `pt2_residual`: the single-pass comparison selector

Both studies use the same solver branch and the same backend:

- Quantum Package backend
- `matvec_backend = "connected_sd_fullq"`
- `method = "ccik_thick"`
- `compression_policy = "fixed_budget"`

The only selector comparison kept here is therefore:

- how the run behaves when determinants are ranked by absolute residual size
- how the run behaves when determinants are ranked by the single-pass PT2-style
  residual score

## Plain-language definitions

`Kprop`

- how many determinants are allowed to generate the next candidate space

`Kstore`

- how many determinants are kept in the next compressed Krylov vector

`mEh`

- millihartree
- `1 mEh = 0.001 Eh`
- values below about `1 mEh` are often treated as chemically accurate

`abs_residual`

- rank determinants by the size of the orthogonalized residual coefficient
- this is the current production default

`pt2_residual`

- rank determinants by a PT2-style score built from the same residual
- in the current code it is a single-pass variant, so it uses one projected
  `H q_k` build per Krylov step and carries the chosen propagation support into
  the next step

## Main conclusion

The production answer in this repository is still:

- use `abs_residual`
- use `ccik_thick`
- keep `fixed_store_keep` moderate
- tune `fixed_propagate_keep` first

The single-pass `pt2_residual` comparison is useful and documented, but the
finished `N2` results do not show a clear win over `abs_residual`.

## Canonical local benchmark folders

The retained local benchmark assets live in:

- `docs/assets/pub/n2_abs_residual_qp_thick/`
- `docs/assets/pub/n2_pt2_residual_single_pass_qp_thick/`

Each folder keeps only the publication-facing files:

- `results.csv`
- `run_metadata.json`
- `progress.json`
- `SUMMARY.md`
- `kprop_scan.png/.pdf`
- `kstore_scan.png/.pdf`

## Selector comparison at a glance

### N2 / 6-31G / R = 2.50 A / CAS(12e,12o)

Reference used:

- `Exact FCI`

Best finished points:

| Selector | Best `Kprop` scan point | Best `Kstore` scan point |
| --- | --- | --- |
| `abs_residual` | `Kprop = 2048`, `Kstore = 65536`, `|ΔE| = 0.000783 mEh` | same point, `|ΔE| = 0.000783 mEh` |
| `pt2_residual` | `Kprop = 1024`, `Kstore = 65536`, `|ΔE| = 0.001028 mEh` | same point, `|ΔE| = 0.001028 mEh` |

Practical readout:

- both selectors are excellent on this system
- `abs_residual` is slightly better
- `pt2_residual` reaches the same accuracy regime without improving on the
  default

### N2 / 6-31G / R = 2.50 A / CAS(14e,14o)

Reference used:

- `Exact FCI`

Best finished points:

| Selector | Best `Kprop` scan point | Best `Kstore` scan point |
| --- | --- | --- |
| `abs_residual` | `Kprop = 1024`, `Kstore = 65536`, `|ΔE| = 0.712209 mEh` | `Kprop = 1024`, `Kstore = 88474`, `|ΔE| = 0.710970 mEh` |
| `pt2_residual` | `Kprop = 1024`, `Kstore = 65536`, `|ΔE| = 0.712209 mEh` | `Kprop = 1024`, `Kstore = 88474`, `|ΔE| = 0.710970 mEh` |

Practical readout:

- on the harder `CAS(14e,14o)` point, the finished single-pass PT2 run matches
  the `abs_residual` benchmark result rather than beating it
- the main tuning lesson stays the same:
  - scan `Kprop` first
  - then fine-tune `Kstore`

## How to read the two plot types

`kprop_scan`

- hold `Kstore` fixed
- change only `Kprop`
- use this plot to find how much propagation space the calculation needs

`kstore_scan`

- hold the best `Kprop` fixed
- change only `Kstore`
- use this plot to see whether the stored vector is already large enough

For the finished `N2` studies, the pattern is consistent:

- `Kprop` is the sharper tuning knob
- `Kstore = 65536` is already a strong starting point
- increasing `Kstore` helps only modestly once the run is already in the right
  regime

## Retained plots

### Production baseline: `abs_residual`

![Abs residual propagate scan](assets/pub/n2_abs_residual_qp_thick/kprop_scan.png)

![Abs residual store scan](assets/pub/n2_abs_residual_qp_thick/kstore_scan.png)

### Comparison variant: single-pass `pt2_residual`

![PT2 residual propagate scan](assets/pub/n2_pt2_residual_single_pass_qp_thick/kprop_scan.png)

![PT2 residual store scan](assets/pub/n2_pt2_residual_single_pass_qp_thick/kstore_scan.png)

## Recommended strategy for larger N2 hard points

If you want a reliable first pass on a similar `N2` active space:

1. use `ccik_thick`
2. use the QP `connected_sd_fullq` backend
3. start with `selector_mode = "abs_residual"`
4. use `compression_policy = "fixed_budget"`
5. start from `fixed_store_keep = 65536`
6. scan `fixed_propagate_keep = 512, 1024, 2048`
7. only then test a larger store budget such as `131072`

That is the current production lesson supported by the finished benchmark data
kept in this repository.
