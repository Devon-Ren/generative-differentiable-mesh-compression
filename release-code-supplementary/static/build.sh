#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
(cd "$ROOT/codec/renderer" && python setup.py build_ext --inplace)
python "$ROOT/postprocess/build_vis.py"
