#!/usr/bin/env bash
# Draco connectivity reference on (1) the original dense Room mesh and (2) the geo-only coarse mesh.
# Needs DracoPy in the active python (pip install DracoPy; if the GPU node has no network, run that one pip line on the login node once).
# v2: no `set -e` (a missing dense codec report made v1 exit silently inside the $(ls ...) substitution).
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
C=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python -c "import DracoPy; print('DracoPy ok')" || pip install -q DracoPy || { echo "DracoPy missing and pip failed; run 'pip install DracoPy' on the login node"; exit 1; }
mkdir -p $W/draco
# dense mesh: our topology stream = top_stream of the 41.4 MB asset's codec report; search the topology run dir for any json that has it
DENSE_REPORT=$(grep -ls '"top_stream"' $(find /scratch/zr2245/room-topology-xmawwL -maxdepth 4 -name '*.json' 2>/dev/null) 2>/dev/null | head -1)
echo "dense codec report: ${DENSE_REPORT:-NOT FOUND (ours row missing for dense; the known value is 30.4 bits/vertex)}"
echo "=== dense mesh ==="
python $HERE/topology_reference.py --mesh $C ${DENSE_REPORT:+--codec-report "$DENSE_REPORT"} --out $W/draco/dense.json 2>&1 | tee $W/draco/dense.log
echo "=== geo-only coarse mesh ==="
python $HERE/topology_reference.py --mesh $W/finetune-geo-only/best.pt --summary $W/deliver-geo-only/summary.json --out $W/draco/geo-only.json 2>&1 | tee $W/draco/geo-only.log
