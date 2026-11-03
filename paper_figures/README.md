# Paper figures (Figs. 3–5 and S1–S3)

Scripts that draw the figures of the paper from the data tables already in this repository. Each script runs
from the repository root without arguments and writes PDF and PNG files to `paper_figures/output/` (not committed).

| Paper figure | Script | Input data (in this repository) | Output |
|---|---|---|---|
| Fig. 3(a), 3(b): iterations vs. squared overlap and gap | `fig3_overlap_gap.py` | `plots/molvqe27_adapt_qrte_dt005_qrte_wmf_qkud_eps005_renyi.csv`, `plots/overlap_gap_iterations.csv` | `fig3a_overlap`, `fig3b_gap` |
| Fig. 4: iterations along the N2 curves | `fig4_figS2_n2pes.py` | `plots/n2_pes_mean_field_pi_over_w_trotter2_vs_qkud_eps0p05_overlap_gap_iterations.csv` | `fig4a_n2pes_singlet`, `fig4b_n2pes_triplet` (panels used in the paper) and `fig4_n2pes_distance` (combined) |
| Fig. 5: iterations vs. Rényi entropy | `fig5_figS3.py` | `plots/molvqe27_adapt_qrte_dt005_qrte_wmf_qkud_eps005_renyi.csv` | `fig5_molvqe27_renyi` |
| Fig. S1: δt and ε scans | `figS1_dt_scan.py` | `paper_figures/data/figS1_dt_eps_scan.csv` (built below) | `figS1_dt_eps_scan` |
| Fig. S2: N2 curves vs. overlap and gap | `fig4_figS2_n2pes.py` | as Fig. 4 | `figS2_n2pes_overlap_gap` |
| Fig. S3: iterations vs. two-body cumulant norm | `fig5_figS3.py` | Fig. 5 table and `plots/qrte_qkud_iterations_vs_2body_cumulant_norm2.csv` | `figS3_cumulant` |
| Fig. 6, Fig. S4, Tables S3–S4 | see `measurement_cost/README.md` | `measurement_cost/results/` | |

Figs. 1 and 2 (exact QRTE and ADAPT-VQE on the hydrogen chains) are not included in this folder.

## Quick start

```bash
python paper_figures/fig3_overlap_gap.py
python paper_figures/fig4_figS2_n2pes.py
python paper_figures/fig5_figS3.py
python paper_figures/figS1_dt_scan.py
```

Requirements: Python 3, NumPy, Matplotlib. `figstyle.py` sets the common style (figures are drawn at their printed
size). The R² values printed by `fig5_figS3.py` are those quoted in the captions of Fig. 5 and Fig. S3.

## Fig. S1 data

Fig. S1 combines two sources for the eight systems of the scan:

- exact QRTE and QRTE-T2 (δt = f·δt_MF): the published scan `plots/qrte_dt_scan_final/qrte_exact_vs_trotter2_loglog.csv`;
- QRTE-T1 (δt = f·δt_MF) and QKUD-T1 (ε = f·δt_MF): 176 runs stored in `data/figS1_t1_qkud_runs.json`, computed by
  `figS1_scan_t1_qkud.py` with the vendored QCANT code, the cached Hamiltonians, 100 iterations, overlap threshold
  10⁻¹⁰ and the 1.6 mHa criterion, as in the published scans.

| Step | Command | Output |
|---|---|---|
| rerun the T1/QKUD points (optional, hours) | `python paper_figures/figS1_scan_t1_qkud.py` | `output/figS1_t1_qkud_runs.json` |
| check a few stored points | `python paper_figures/figS1_scan_t1_qkud.py --compare paper_figures/data/figS1_t1_qkud_runs.json --limit 3` | prints OK/MISMATCH |
| merge both sources | `python paper_figures/figS1_build_table.py` | `output/figS1_dt_eps_scan.csv` (identical to `data/figS1_dt_eps_scan.csv`) |
| draw | `python paper_figures/figS1_dt_scan.py` | `output/figS1_dt_eps_scan.pdf` |

`figS1_scan_t1_qkud.py` needs Qulacs and SciPy in addition. `--fast` computes the projected Hamiltonian from one
sparse product per vector (same Krylov vectors and eigenproblem); `--validate` reruns four published exact/T2
points. QCANT's `_version.py` is generated on installation and is not in git; the script supplies it, or run
`pip install -e vendor/QCANT`.

## Check

From a fresh clone, all seven figure files above are pixel-identical to those in the manuscript, and
`figS1_build_table.py` rebuilds `data/figS1_dt_eps_scan.csv` exactly.
