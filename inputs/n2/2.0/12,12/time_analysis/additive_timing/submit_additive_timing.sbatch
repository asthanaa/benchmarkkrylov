#!/usr/bin/env bash
#SBATCH --job-name=n2_12_12_additive_timing
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

ROOT="/home/ayush.asthana/code/CompressK"
CASE_DIR="${ROOT}/inputs/n2/2.0/12,12/time_analysis/additive_timing"
BASE_SETUP="${ROOT}/inputs/n2/2.0/12,12/system_setup"
QP_ROOT="/home/ayush.asthana/quantum_package/qp2"
OPENBLAS_LIB="/cm/shared/apps/spack/opt/spack/linux-rhel9-icelake/gcc-11.3.1/openblas-0.3.27-tjsex4shr6vfowj3tgp23jyvftvaxl2q/lib"

cd "${ROOT}"

echo "additive-timing start $(date -Is)"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "job=${SLURM_JOB_ID:-unknown}"
echo "allocated_cpus=${SLURM_CPUS_PER_TASK:-unknown}"

source "${ROOT}/scripts/advanced/qp_slurm_env.sh"
set +e
set +u
set +o pipefail
source "${QP_ROOT}/quantum_package.rc" >/dev/null 2>&1
rc=$?
set -o pipefail
set -u
set -e
if [[ ${rc} -ne 0 ]]; then
  echo "failed to source ${QP_ROOT}/quantum_package.rc (rc=${rc})" >&2
  exit "${rc}"
fi
export LD_LIBRARY_PATH="${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cka_export_qp_maxmem_from_slurm

echo "rebuild ccik plugin start $(date -Is)"
python "${ROOT}/scripts/advanced/qp_ccik_bridge.py" rebuild-plugin --repo "${ROOT}/ccik_qp/stable/ccik"
echo "rebuild ccik plugin done $(date -Is)"

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
    ln -s "../../../system_setup" "${run_dir}/system_setup"
  fi

  echo "additive-timing cores=${cores} start $(date -Is)"
  /usr/bin/time -v bash "${ROOT}/scripts/advanced/run_n2_thick_slurm.sh" "${input_json}" --force
  echo "additive-timing cores=${cores} done $(date -Is)"

  python "${ROOT}/scripts/advanced/analyze_n2_additive_timing.py" --root "${CASE_DIR}" || true
done

python "${ROOT}/scripts/advanced/analyze_n2_additive_timing.py" --root "${CASE_DIR}"
echo "additive-timing done $(date -Is)"
