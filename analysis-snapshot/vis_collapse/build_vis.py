"""Compile native/native_qem_vis.cpp -> native/_native_qem_vis<EXT_SUFFIX>  (CPU only, ~1 min).
Same recipe as stage2/build_legacy.py.  Run inside the pmsh-cu126 + .venv environment:
    python build_vis.py
"""
import os, shlex, subprocess, sys, sysconfig
from pathlib import Path
try:
    import pybind11; include = pybind11.get_include()
except ImportError:
    import torch; include = str(Path(torch.__file__).parent / 'include')
root = Path(__file__).resolve().parent / 'native'
out = root / ('_native_qem_vis' + sysconfig.get_config_var('EXT_SUFFIX'))
cmd = shlex.split(os.environ.get('CXX', 'c++')) + ['-O3', '-std=c++17', '-shared', '-fPIC', '-I' + include, '-I' + sysconfig.get_path('include'),
                                                   str(root / 'native_qem_vis.cpp'), '-o', str(out)]
if sys.platform == 'darwin': cmd += ['-undefined', 'dynamic_lookup']
subprocess.run(cmd, check=True); print('built', out)
