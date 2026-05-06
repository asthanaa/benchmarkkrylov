#!/usr/bin/env bash
#SBATCH --job-name=n2_12_12_time_analysis
#SBATCH --partition=talon-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

ROOT="/home/ayush.asthana/code/CompressK"
CASE_DIR="${ROOT}/inputs/n2/2.0/12,12/time_analysis"
BASE_SETUP="${ROOT}/inputs/n2/2.0/12,12/system_setup"

cd "${ROOT}"

echo "time-analysis start $(date -Is)"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "job=${SLURM_JOB_ID:-unknown}"
echo "allocated_cpus=${SLURM_CPUS_PER_TASK:-unknown}"

for cores in 02 04 08; do
  run_dir="${CASE_DIR}/cores${cores}"
  input_json="${run_dir}/input.json"
  if [[ ! -f "${input_json}" ]]; then
    echo "missing input: ${input_json}" >&2
    exit 2
  fi
  if [[ ! -d "${BASE_SETUP}" ]]; then
    echo "missing shared setup: ${BASE_SETUP}" >&2
    exit 2
  fi
  if [[ ! -e "${run_dir}/system_setup" ]]; then
    ln -s "../../system_setup" "${run_dir}/system_setup"
  fi

  echo "time-analysis cores=${cores} start $(date -Is)"
  /usr/bin/time -v bash "${ROOT}/scripts/advanced/run_n2_thick_slurm.sh" "${input_json}"
  echo "time-analysis cores=${cores} done $(date -Is)"

  python "${ROOT}/scripts/advanced/analyze_n2_time_analysis.py" --root "${CASE_DIR}" || true
done

python "${ROOT}/scripts/advanced/analyze_n2_time_analysis.py" --root "${CASE_DIR}"
echo "time-analysis done $(date -Is)"
