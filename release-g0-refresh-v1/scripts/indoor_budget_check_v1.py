import sys, os
LAB = '/scratch/zr2245/meshsplatting-original-lossless-v1'; up = os.path.join(LAB, 'upstream'); sys.path.insert(0, up); os.chdir(up)
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, update_indoor
p = ArgumentParser(); lp = ModelParams(p); op = OptimizationParams(p); pp = PipelineParams(p); p.add_argument('--indoor', action='store_true', default=False)
args = p.parse_args(['-s', '/tmp/x', '-m', '/tmp/y', '--indoor', '--max_points', '123456'])
ops = op.extract(args); before = (ops.max_points, ops.add_percentage); ops = update_indoor(ops); after = (ops.max_points, ops.add_percentage)
print(f'after extract: max_points {before[0]} add_percentage {before[1]} | after update_indoor: max_points {after[0]} add_percentage {after[1]}')
ok = (after[0] == 123456)
print('INDOOR BUDGET CHECK', 'PASS: --max_points survives --indoor (add_percentage is the one the preset overwrites)' if ok else 'FAIL: update_indoor changed max_points')
sys.exit(0 if ok else 1)
