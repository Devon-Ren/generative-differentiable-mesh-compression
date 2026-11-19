import argparse
import sys
import tempfile
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stage2', default=str(Path(__file__).resolve().parent / 'codec'))
    a = p.parse_args()
    stage2 = Path(a.stage2).resolve()
    for d in (stage2, stage2 / 'codec'):
        sys.path.insert(0, str(d))
    import delivery
    import topology
    from appearance import expand_wire
    with tempfile.TemporaryDirectory() as work:
        decoded = topology.read_asset(a.asset, Path(work) / 'native', Path(work))
    state = delivery.restore(decoded)
    mesh = {k: v for k, v in state.items() if not k.startswith('app_')}
    mesh['features_rest'] = torch.from_numpy(expand_wire(state))
    torch.save(mesh, a.out)
    print('vertices', len(mesh['triangles_points']), 'faces', len(mesh['_triangle_indices']))


if __name__ == '__main__':
    main()
