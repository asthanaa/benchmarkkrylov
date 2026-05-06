# N2 CAS(12,12) Timing Analysis Summary

Complete runs: 6 / 6

| cores | method | wall s | final abs mHa | min abs mHa | HQ CPU s | select | sort | internal | external | assign |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | baseline | 783.570 | 0.470 | 0.470 | 873.774 | 40.960 | 0.000 | 13.197 | 819.375 | 0.242 |
| 2 | secondortho | 791.906 | 0.108 | 0.108 | 872.839 | 40.861 | 0.000 | 12.974 | 818.763 | 0.241 |
| 4 | baseline | 579.409 | 0.470 | 0.470 | 909.923 | 44.424 | 0.000 | 13.590 | 851.666 | 0.244 |
| 4 | secondortho | 637.684 | 0.108 | 0.108 | 973.693 | 50.248 | 0.000 | 14.236 | 908.926 | 0.282 |
| 8 | baseline | 583.129 | 0.470 | 0.470 | 1115.533 | 70.780 | 0.000 | 16.244 | 1028.161 | 0.349 |
| 8 | secondortho | 592.561 | 0.108 | 0.108 | 1116.789 | 70.299 | 0.000 | 16.009 | 1030.127 | 0.354 |

## Wall-Time Scaling

Speedup values are relative to the 2-core run of the same method.

- baseline: 4 cores = 1.352360259978194, 8 cores = 1.3437320910659447
- secondortho: 4 cores = 1.2418466548511669, 8 cores = 1.336413242757447

## Second-Orthogonalization Overhead

Values are `secondortho wall_s / baseline wall_s` at the same core count.

- 2 cores: 1.0106385615015598
- 4 cores: 1.1005766472351128
- 8 cores: 1.0161733093547662

## Notes

- `wall_s` comes from the Python runner around the QP CCIK executable.
- `HQ CPU s` components come from QP verbose timing lines and are CPU seconds, not wall seconds.
- The HQ components show where the CPU work is concentrated; wall-speedup is the primary scaling metric.
