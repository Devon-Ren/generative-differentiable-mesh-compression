"""Compile native/index_ctx.cpp -> native/_index_ctx<EXT_SUFFIX>.  Needs the stage2 package for its codec/native.cpp:
    bash stage2_run.sh build_index_ctx.py            (stage2_run.sh passes --stage2 <PKG>)
    python build_index_ctx.py --stage2 /scratch/zr2245/shared-collapse-stage2-v2
"""
import argparse, os, shlex, subprocess, sys, sysconfig
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2'); a = p.parse_args()
codec = Path(a.stage2).resolve() / 'codec'
if not (codec / 'native.cpp').exists(): raise SystemExit(f'{codec}/native.cpp not found: pass --stage2 <stage2 package dir>')
try:
    import pybind11; include = pybind11.get_include()
except ImportError:
    import torch; include = str(Path(torch.__file__).parent / 'include')
root = Path(__file__).resolve().parent / 'native'
out = root / ('_index_ctx' + sysconfig.get_config_var('EXT_SUFFIX'))
cmd = shlex.split(os.environ.get('CXX', 'c++')) + ['-O3', '-std=c++17', '-shared', '-fPIC', '-I' + include, '-I' + sysconfig.get_path('include'), '-I' + str(codec),
                                                   str(root / 'index_ctx.cpp'), '-o', str(out)]
if sys.platform == 'darwin': cmd += ['-undefined', 'dynamic_lookup']
subprocess.run(cmd, check=True); print('built', out)
