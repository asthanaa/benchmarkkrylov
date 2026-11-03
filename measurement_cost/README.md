# Measurement-cost analysis (Fig. 6, Table I, Fig. S4, Table S3)

This folder contains the code and the stored results behind the measurement-cost comparison of QRTE-T2 and
ADAPT-VQE in the paper:

| Paper item | What it shows | Built by | Stored result |
|---|---|---|---|
| Fig. 6 | shots to chemical accuracy vs. Rényi entropy | `build_figures.py` | `results/fig6_shots.pdf` (generated) |
| Fig. S4 | CNOT count of the deepest circuit vs. Rényi entropy | `circuit_depth.py` | `results/figS4_depth.pdf` (generated) |
| Table S3 | per-system iterations, shots, deepest-circuit CNOTs | `circuit_depth.py` | `results/table_s3.tex` |
| Table I (main text; formerly Table S4) | STO-3G hydrogen chains H4–H9 ordered by qubit count, with power-law fits | `table_s4_hchains.py` | `results/table_s4.tex` |
| all numbers | one row per system | `build_figures.py`, `circuit_depth.py` | `results/measurement.csv`, `results/depth.csv` |

## Quick start: rebuild the figures and the tables (seconds)

```bash
cd measurement_cost
python build_figures.py      # Fig. 6, measurement.csv
python circuit_depth.py     # Fig. S4, Table S3, depth.csv
python table_s4_hchains.py  # Table I (main text)
```

This needs only the files in `results/` (no quantum simulation). The full pipeline below regenerates those files.

## Full pipeline

Run from `measurement_cost/`, in the benchmark environment (Qulacs, NumPy, SciPy, Matplotlib; `openpyxl` for the
optional workbook). The vendored QCANT package and the cached Hamiltonians of the repository are used directly (QCANT's
`_version.py`, generated on installation and not in git, is supplied by `paths.py`; alternatively run `pip install -e ../vendor/QCANT`).

| Step | Command | Output | Run time |
|---|---|---|---|
| 1 | `python resources.py` | `results/resources.csv`: Pauli terms, λ, CNOTs of one Trotter step | minutes |
| 2 | `python krylov_bases.py W all` | `work/W/<case>.npz`: Krylov matrices **S**, **H** with δt = π/W | minutes (small) to hours (18–20 qubits) |
| 2b | `EXTRA=100 python krylov_bases.py W <cases>` | extended bases (100 vectors) for the systems listed below | hours |
| 2c | `python krylov_bases.py MF all` | `work/MF/`: δt = π/W_MF (Fig. 5 protocol), for Table S3 | as step 2 |
| 3 | `python shot_allocation.py W` and `python shot_allocation.py MF` | `results/alloc_W.csv`, `results/alloc_MF.csv`: shots (Monte Carlo) | minutes to ~1 h per system |
| 4 | `python adapt_lower_bound.py` | `results/adapt_lb.csv`: ADAPT-VQE evaluations E_A, final-ansatz CNOTs (and the unused CNOT-weighted evaluations G_A) | seconds |
| 5 | `python gate_volume.py` | `results/gate_volume.csv`: CNOTs of one controlled / uncontrolled Trotter step (C_c, C_u) and per-shot counts of the Fig. 6 estimate | seconds |
| 6 | `python build_figures.py` | Fig. 6, `results/measurement.csv` | seconds |
| 7 | `python circuit_depth.py` | Fig. S4, Table S3, `results/depth.csv` | seconds |
| 8 | `python table_s4_hchains.py` | `results/table_s4.tex` (Table I of the main text) | seconds |

`work/` (Krylov matrices, about 6 MB) is not committed; steps 2–3 regenerate it.

Bases were extended to 100 vectors (step 2b) when the shot-optimal dimension was the largest one available:
- δt = π/W: `BeH2_equi_cc_pVTZ_2`, `CO_equi_cc_pVTZ_2`, `N2_equi_cc_pVTZ_2`, `H7_near_equi_sto3g_1`,
  `H8_near_equi_sto3g_0`, `H9_near_equi_sto3g_1`, `fenoco3_minus`, `FeO`, `feno_2plus`
- δt = π/W_MF: `CO_equi_cc_pVTZ_2`, `N2_equi_cc_pVTZ_2`, `H6_near_equi_sto3g_0`, `H9_near_equi_sto3g_1`

The column `vectors` of `results/alloc_*.csv` gives the basis size used for each system.

## Optional checks

| Command | What it checks |
|---|---|
| `python audit_fig5.py` | after step 2c: the rebuilt δt = π/W_MF runs reproduce every Fig. 5 iteration count (`results/audit_fig5_WMF.csv`; 27/27 identical, energies within 1e-8 Ha except late, ill-conditioned FeO/FeNO2+ iterations, 1e-5 Ha) |
| `python check_against_qrte_qulacs.py W <cases>` | the unmodified production function `qrte_qulacs` gives the same energy history as `krylov_bases.py` |
| `python gate_volume.py --circuits H4_near_equi_sto3g_0` then `python crosscheck_workbook.py` | `results/measurement_crosscheck.xlsx`: every factor of every system as live spreadsheet formulas, every circuit of the H4 estimate, and every ADAPT-VQE iteration |

## How things are counted

**Shots, both methods.** The Pauli terms c_k P_k of the qubit Hamiltonian (identity excluded) are sampled with
probabilities |c_k|/λ, λ = Σ|c_k|, without measurement grouping. One energy evaluation with standard deviation
σ = 1.6 mHa costs λ²/σ² shots. Both methods must reach a noiseless error ≤ 1.6 mHa, with shot noise σ.

**QRTE-T2** (δt = π/W, W = exact width of the (N_α, N_β) sector, `results/widths/`; one second-order Trotter step U
per time step; Krylov vectors |φ_j⟩ = U^j|Φ_ref⟩):
- **H** is not Toeplitz with a Trotter step, so every element ⟨φ_j|H|φ_l⟩ (j ≤ l) is measured: diagonal, one circuit
  (U^j, then the sampled Pauli); off-diagonal, real and imaginary Hadamard tests (U^j, then controlled U^(l−j)).
  **S** is Toeplitz: real and imaginary Hadamard tests of controlled U^m.
- Noise: Gaussian with variance λ²/n (**H**) or 1/n (**S**) for n shots of a circuit.
- Shots per circuit ∝ σ_c|∂E/∂x_c| (first-order energy sensitivity, floor 1e-2 of the largest share) or uniform,
  whichever needs fewer; energy by canonical orthogonalization with relative threshold
  τ ∈ {1e-10, 10^-9.5, …, 10^-0.5}. The basis dimension and τ with the fewest total shots are selected among those
  whose noiseless energy is within 1.6 mHa and whose energy standard deviation over 400 noise samples is ≤ σ.

**Circuit depth (Fig. S4, Table S3, Table I)** = CNOT count of the deepest circuit:
- Controlled Pauli rotation of weight w: 2w CNOTs (only the central Rz is controlled); no cancellation between
  terms. One second-order Trotter step = 2L rotations for L Pauli terms (exact symmetric step: 2L−1, < 0.6 % fewer).
  This gives C_c CNOTs for one controlled step (`gate_volume.csv`).
- QRTE-T2: the deepest circuit is the Hadamard test of ⟨Φ|U^k|Φ⟩, with k·C_c CNOTs, for k = the shot-optimal basis
  dimension of the Fig. 6 estimate (`kK` in `measurement.csv`; Table S3, Table I, and filled markers of Fig. S4) or k = K,
  the iteration of convergence (open markers of Fig. S4).
- Not counted: the controlled Pauli string that ends each Hadamard test of H (≤ N two-qubit gates) and single-qubit
  gates. The QRTE-T2 counts assume one Pauli rotation per Hamiltonian term without compilation and are upper
  estimates.
- ADAPT-VQE: the final ansatz, with the compiled CNOT count c_{K_A} of Hagelueken et al. (`cnot_final` in
  `adapt_lb.csv`).

The earlier "gate executions" (CNOTs × shots) quantities GK and GA are still computed (`gate_volume.csv`,
`adapt_lb.csv`, `measurement.csv`) but are not used in the paper.

**ADAPT-VQE lower bound** (TETRIS-ADAPT-VQE records of Hagelueken et al., Zenodo 19051536, in `dataset/`), per
iteration k = 1…K_A, with N qubits, p_k parameters and c_k CNOTs (c_0 = 0):
- 4N energy evaluations for the gradients of all pool operators, measured in commuting groups (Anastasiou et al.,
  arXiv:2306.03227), on the ansatz of iteration k−1; 1 energy evaluation and 2p_k parameter-shift evaluations on the
  ansatz of iteration k.
- Shots NA = (λ²/σ²) Σ_k (4N + 1 + 2p_k) (the unused GA = (λ²/σ²) Σ_k [4N c_(k−1) + (1 + 2p_k) c_k] is also stored).
- Lower bound: real optimizers need more evaluations, and noisy gradients can add iterations.

H10 is not in Table S3 because ADAPT-VQE did not reach chemical accuracy in the data of Hagelueken et al. The CO
triplet uses the MolVQE-matching Hamiltonian of the Fig. 5 runs (`hamiltonian_cache/co_spin2_molvqe_matching/`),
selected through the Fig. 5 data table `plots/molvqe27_adapt_qrte_dt005_qrte_wmf_qkud_eps005_renyi.csv`.

## Files

| File | Content |
|---|---|
| `paths.py` | shared paths and the Hamiltonian loader (same payload as the Fig. 5 runs) |
| `figstyle.py` | figure style |
| `results/widths/<case>.json` | exact sector width W = E_max − E_ref (Lanczos), W_MF, Hamiltonian cache used |
| `results/resources.csv` | Pauli terms, mean weight, λ, CNOTs of a controlled / uncontrolled first-order step |
| `results/alloc_W.csv`, `results/alloc_MF.csv` | Monte Carlo results; `mc_*` = shot-optimal estimate of Fig. 6 (`mc_shots`), `mcg_*` = gate-weighted variant (not used in the paper) |
| `results/adapt_lb.csv` | ADAPT-VQE K_A, E_A, G_A (and their pool / optimization parts) |
| `results/gate_volume.csv` | C_c, C_u, circuits, CNOTs per shot and gate executions GK (unused) of the Fig. 6 estimate |
| `results/depth.csv` | deepest-circuit CNOTs: QRTE-T2 at K and at the Fig. 6 basis, ADAPT-VQE final ansatz |
| `results/measurement.csv` | everything plotted in Fig. 6 and listed in Table S3 and Table I (shots, iterations) |
| `results/table_s3.tex`, `results/table_s4.tex` | Table S3 and Table I of the main text (file name kept), from `depth.csv` |
| `results/circuits_H4_near_equi_sto3g_0.csv` | every circuit and shot weight of the H4 estimate |
