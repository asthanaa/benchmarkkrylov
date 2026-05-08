# Asymmetric CK -> CKCI-0 Probe

- target error: `1.0e-08`
- CKCI-0 pool cap: `300000` determinants
- current compact CKCI-n best: `37483` determinants, error `3.732694e-09`
- older top-512 path: `130013` determinants, error `3.732438e-09`

## Best CK

- run: `d17_kp3000_ks500000_nr5_cy16`
- determinant count: `7555`
- absolute error: `3.862974e-01`

## Best CKCI-0

- run: `d17_kp3000_ks500000_nr5_cy16`
- determinant count: `7555`
- absolute error: `2.523299e-01`
- nnz: `129649`

## Best CKCI-n

- stage: `CKCI-2`
- run: `d17_kp2000_ks50000_nr5_cy16`
- determinant count: `36779`
- absolute error: `2.000000e-02`
- nnz: `747685`

## Recommendation

Asymmetric CKCI-n did not beat the existing compact high-K CKCI-n strategy.

## Output Files

- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/asym_ckci0_probe/data/asymmetric_ck_runs.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/asym_ckci0_probe/data/ckci0_runs.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/asym_ckci0_probe/data/ckci_n_runs.csv`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/asym_ckci0_probe/data/asymmetric_ckci0_summary.json`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/asym_ckci0_probe/asymmetric_ckci0_probe.pdf`
- `/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/asym_ckci0_probe/asymmetric_ckci0_probe.png`
