"""Debug script: set up N2 EZFIO in a persistent directory and run ccik directly."""
import sys, subprocess, shlex
from pathlib import Path

sys.path.insert(0, "scripts/_shared"); sys.path.insert(0, "src")
from dataclasses import replace as dc_replace
from qp_ccik_common import (
    discover_qp_paths, run_qp_setup, set_ccik_params,
    CCIKParams, fixed_budget_bridge_kwargs, write_ezfio_value
)

qp = discover_qp_paths()
local_ccik = qp.qp_root / "plugins" / "local" / "ccik" / "ccik"
if local_ccik.exists(): qp = dc_replace(qp, ccik_binary=local_ccik)
print(f"ccik binary: {qp.ccik_binary}  exists={qp.ccik_binary.exists()}")

wd  = Path("/tmp/n2_ccik_debug")
wd.mkdir(exist_ok=True)
xyz   = wd / "n2.xyz"
ezfio = wd / "n2.ezfio"
xyz.write_text("2\nN2\nN  0.0  0.0  0.0\nN  0.0  0.0  1.10\n")

if not ezfio.exists():
    print("Running QP setup …")
    run_qp_setup(qp, basis="6-31g", ezfio=ezfio, xyz=xyz, ncore=1, ncas=12, nmo_total=18)
    print("Setup done.")
else:
    print("Reusing existing EZFIO.")

bk = fixed_budget_bridge_kwargs(fixed_store_keep=65_536, fixed_propagate_keep=512, external_keep=2048)
params = CCIKParams(
    **bk,
    ccik_max_cycle=2, ccik_restart_m_cycle=3, ccik_restart_nroot=2,
    ccik_verbose=True, ccik_matvec_backend="connected_sd_fullq",
    ccik_selector_mode="abs_residual",
)
set_ccik_params(qp, ezfio, params)
print("Params written. Running ccik …")

rc   = shlex.quote(str(qp.quantum_package_rc))
ez   = shlex.quote(str(ezfio))
cb   = shlex.quote(str(qp.ccik_binary))
cmd  = f"source {rc} && qp set_file {ez} && {cb} {ez}"
result = subprocess.run(["/bin/bash", "-lc", cmd], text=True)
print(f"\nExit code: {result.returncode}")
