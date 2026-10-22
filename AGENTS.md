# Plotting source preference

For every future plot covering the full 27-system MolVQE benchmark, use the
new MolVQE-matching calculations for `CO_equi_cc_pVTZ_2`. This is an explicit
user preference. Select the run matching the plotted method and parameters:

| Calculation | CO spin 2 result directory | Chemical-accuracy hit iteration |
| --- | --- | ---: |
| QRTE, dt=0.05, first-order Trotter, one slice | `runs/co_spin2_molvqe_matching/qrte_dt0p05_trotter1/CO_equi_cc_pVTZ_2/` | 80 |
| QRTE, dt=pi/W_MF, second-order Trotter, one slice | `runs/co_spin2_molvqe_matching/qrte_pi_over_wmf_trotter2/CO_equi_cc_pVTZ_2/` | 12 |
| QKUD, epsilon=0.05, first-order Trotter, one slice | `runs/co_spin2_molvqe_matching/qkud_eps0p05_trotter1/CO_equi_cc_pVTZ_2/` | 9 |

Read and validate the actual `summary.json` and `min_energy_history.csv` when
plotting; the counts above describe the current completed runs. The new CO
triplet reference is -112.60166830790772 Eh, with 1.6 mHa chemical accuracy.
Its Hamiltonian is
`hamiltonian_cache/co_spin2_molvqe_matching/CO_equi_cc_pVTZ_2.npz`;
reference provenance is in `runs/co_spin2_molvqe_matching/reference/reference.json`.

Do not use the earlier CO spin 2 runs or its lower-energy (-112.656390169659 Eh)
cache as the CO point in these plots. Existing collectors and the original
27-system cache manifest may still select those older sources, so override
their CO selection explicitly. Any Hamiltonian-derived CO descriptors plotted
with these new results must come from the matching new Hamiltonian. Record
the selected source in plot CSV/provenance outputs. If the requested method
or parameters lack a matching new run, flag that gap instead of substituting
an older CO result or a run with different settings.
