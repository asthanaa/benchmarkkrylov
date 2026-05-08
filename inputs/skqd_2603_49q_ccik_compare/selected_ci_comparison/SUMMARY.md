# Local HCI/CIPSI Selected-CI Comparison

This folder contains locally computed heat-bath CI and CIPSI-style variational selected-CI curves for the SKQD 49Q_v1 public Hamiltonian.

- selected-CI budgets: `[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]`
- target error: `1.0e-08`

- best Heat-bath CI: `131072` determinants, error `2.522441e-01`
- best CIPSI: `131072` determinants, error `2.522441e-01`
- best CKCI-n reference: `37483` determinants, error `3.732694e-09`

## Outputs

- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_comparison/data/selected_ci_runs.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_comparison/data/selected_ci_summary.json`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_comparison/pub/skqd_ckci_hci_cipsi_comparison.pdf`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_comparison/pub/skqd_ckci_hci_cipsi_comparison.png`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_comparison/pub/skqd_ckci_hci_cipsi_comparison_caption.tex`

## Caption

Comparison of local selected-CI baselines with CK and CKCI-$n$ for the public 49Q\_v1 synthetic Hamiltonian. Heat-bath CI and CIPSI curves were generated in this work from deterministic variational selected-CI calculations on the same public Hamiltonian and initial determinant; the HCI selection ranks external determinants by $\max_i |H_{ai} c_i|$, while the CIPSI selection uses an Epstein--Nesbet score $|\sum_i H_{ai} c_i|^2/|E-H_{aa}|$. The best HCI point shown uses 131072 determinants with absolute error $2.52 \times 10^{-1}$; the best CIPSI point uses 131072 determinants with absolute error $2.52 \times 10^{-1}$. The CKCI-$n$ reference path reaches error $3.73 \times 10^{-9}$ at 37483 determinants. Teal diamonds reproduce the published experimental SKQD points, with zero-error points clipped only for display on the logarithmic axis.
