# N2 CAS(12,12) Second-Orthogonalization Timing Analysis

This folder contains a controlled Talon timing campaign for comparing the
standard fixed-budget QP CCIK run with the second-orthogonalization variant.

The campaign uses:

- `Kprop = 500`
- `Kstore = 10000`
- `m = 31`
- `nroot = 3`
- `ncycles = 3`
- `OMP threads = 2, 4, 8`

The SLURM campaign requests 8 cores once and runs the 2-, 4-, and 8-core cases
sequentially on the same allocated node. This keeps node-level variability out
of the CPU-scaling comparison.

Run from Talon:

```bash
cd /home/ayush.asthana/code/CompressK/inputs/n2/2.0/12,12/time_analysis
sbatch submit_time_analysis.sbatch
```

After jobs finish, summarize timings with:

```bash
python /home/ayush.asthana/code/CompressK/scripts/advanced/analyze_n2_time_analysis.py \
  --root /home/ayush.asthana/code/CompressK/inputs/n2/2.0/12,12/time_analysis
```
