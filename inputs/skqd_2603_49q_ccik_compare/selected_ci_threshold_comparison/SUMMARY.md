# Threshold HCI/CIPSI Comparison

This campaign locally reproduces threshold-based HCI and CIPSI selected-CI baselines for the SKQD 49Q_v1 Hamiltonian.

- CIPSI epsilon sweep: `1.0e-04` to `1.0e-17` with `31` values
- HCI epsilon sweep: `1.0e-05` to `1.0e-20` with `31` values
- max iterations: `30`
- safety pool cap: `300000` determinants; this is a stop condition, not a truncation

- best HCI threshold sweep: epsilon `3.162e-08`, `202383` determinants, error `9.999996e-03`
- best CIPSI threshold sweep: epsilon `2.512e-07`, `18027` determinants, error `9.999996e-03`
- best CKCI-n reference: `37483` determinants, error `3.732694e-09`

## Outputs

- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/data/threshold_iteration_trace.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/data/threshold_run_summary.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/data/threshold_envelope.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/data/threshold_summary.json`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/pub/skqd_ckci_hci_cipsi_threshold_comparison.pdf`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/pub/skqd_ckci_hci_cipsi_threshold_comparison.png`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_threshold_comparison/pub/skqd_ckci_hci_cipsi_threshold_comparison_caption.tex`

## Caption

Comparison of CKCI-$n$ against locally reproduced threshold-based HCI and CIPSI selected-CI baselines for the public 49Q\_v1 synthetic Hamiltonian. HCI and CIPSI were run with the paper's threshold-selection rules, retaining all accepted configurations and stopping only at natural termination, the iteration cap, or the safety pool cap. The best HCI threshold point uses 202383 determinants with absolute error $1.00 \times 10^{-2}$; the best CIPSI threshold point uses 18027 determinants with absolute error $1.00 \times 10^{-2}$. The CKCI-$n$ path reaches error $3.73 \times 10^{-9}$ at 37483 determinants. Teal diamonds show the experimental SKQD points digitized from the paper.
