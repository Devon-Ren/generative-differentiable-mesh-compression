"""make_normals_input_v1.py (VERSION v1 2026-11-04): NeRF-Synthetic train PNGs (RGBA) -> <scene>/images/<name>.jpg (composited over black,
what upstream extract_normals.py expects: jpg files in an 'images' folder), and after Metric3D ran: mask <scene>/normals/<name>.png with the
PNG alpha -> <scene>/normals_masked/<name>.png (background = encoded zero vector 128,128,128, so the normal prior is silent there).

  python make_normals_input_v1.py --scene DIR --stage jpg      # before extract_normals.py
  python make_normals_input_v1.py --scene DIR --stage mask     # after  extract_normals.py -s DIR -i images
"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image

VERSION = 'make_normals_input_v1 2026-11-04'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); ap.add_argument('--stage', choices=['jpg', 'mask'], required=True)
    ap.add_argument('--alpha_thresh', type=int, default=1)
    a = ap.parse_args(); scene = Path(a.scene); train = scene / 'train'
    pngs = sorted(p for p in train.glob('*.png') if '_depth' not in p.name and '_normal' not in p.name)
    if not pngs:
        sys.exit(f'FAILED: no train PNGs in {train}')
    if a.stage == 'jpg':
        out = scene / 'images'; out.mkdir(exist_ok=True); n = 0
        for p in pngs:
            im = np.array(Image.open(p).convert('RGBA')).astype(np.float32) / 255.0
            rgb = im[..., :3] * im[..., 3:4]  # over black
            Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out / (p.stem + '.jpg'), quality=97); n += 1
        print(f'[normals-input] {VERSION}: wrote {n} jpg to {out}')
    else:
        src = scene / 'normals'; out = scene / 'normals_masked'; out.mkdir(exist_ok=True); n, miss = 0, []
        for p in pngs:
            q = src / (p.stem + '.png')
            if not q.is_file():
                miss.append(p.stem); continue
            alpha = np.array(Image.open(p).convert('RGBA'))[..., 3]
            nm = np.array(Image.open(q).convert('RGB'))
            if nm.shape[:2] != alpha.shape:
                sys.exit(f'FAILED: {q} is {nm.shape[1]}x{nm.shape[0]}, image is {alpha.shape[1]}x{alpha.shape[0]}')
            nm[alpha < a.alpha_thresh] = 128
            Image.fromarray(nm).save(out / (p.stem + '.png')); n += 1
        if miss:
            sys.exit(f'FAILED: {len(miss)} normal maps missing in {src}, e.g. {miss[:5]}')
        print(f'[normals-input] {VERSION}: wrote {n} masked normal maps to {out}')


if __name__ == '__main__':
    main()
