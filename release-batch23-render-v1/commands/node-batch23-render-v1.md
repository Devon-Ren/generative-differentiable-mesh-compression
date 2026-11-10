# node-batch23-render-v1 (2026-11-10): qualitative renders for the figure, on the reserved node

Six small GPU jobs (5–15 min each, `--reservation`, no fairshare). Each decodes the delivered assets of one scene with the real stage-2 decoder, renders two test views of every asset next to the ground truth and the FP32 original at the metrics' 4× supersampling, and writes `montage.png` + per-view PNGs + `psnr.txt` under `/scratch/zr2245/renders-b23/<scene>/`.

| job | columns (left → right) | why |
|---|---|---|
| room, garden, truck, drjohnson | GT, FP32, controller 20 / 10 / 5 MB (v9b) | the main qualitative figure: one indoor, one outdoor, one T&T, one Deep Blending |
| lego | GT, FP32, controller 6 / 3 / 1.5 MB | NeRF-Synthetic row |
| room-g0 | GT, FP32, λ-grown 5 MB, train-at-budget 486k, post-hoc collapse 486k | the motivation figure: same vertex count, three ways |

Error maps (|render − GT| × 4) are the second row of each view. If the decoder call fails on an asset, the job falls back to that asset's pre-quantisation mesh and labels the column `NAME~`; the log says so.

## §1 write files, check, submit (one paste)

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/render
cat > $AN/render/decode_asset_v1.py <<'PYEOF_DEC1'
"""decode_asset_v1.py (VERSION decode_asset_v1 2026-11-10): decode a delivered asset (deliver/delivery/asset.mslb) with the stage-2 decoder and write a plain
MeshSplatting-style checkpoint (triangles_points, _triangle_indices, vertex_weight, features_dc, features_rest) that render_ckpt_v1.py / eval_ckpt_v1.py can render.
Run inside the stage-2 env:  bash $PKG/stage2_run.sh decode_asset_v1.py --deliver-dir <...>/axisB/v1.0/deliver --out <dir>/decoded.pt
(stage2_run.sh prepends --stage2 <PKG>).  Uses the delivery's own compiled native coder (delivery/native) as the cache.  If the decode fails, falls
back to shared-final.pt (the fitted shared mesh before quantisation and entropy coding) and writes <out>-prequant.pt instead, with a WARNING.
"""
import argparse, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import torch

VERSION = 'decode_asset_v1 2026-11-10'


def T(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    return torch.as_tensor(np.asarray(x))


def to_ckpt(w, n_hint=None):
    xyz = T(w['triangles_points']).float().reshape(-1, 3); n = xyz.shape[0]
    faces = T(w['_triangle_indices']).to(torch.int32).reshape(-1, 3)
    dc = T(w['features_dc']).float().reshape(n, 1, 3)
    vw = T(w['vertex_weight']).float().reshape(-1)
    if vw.numel() == 1: vw = vw.expand(n).clone()
    assert vw.numel() == n, f'vertex_weight has {vw.numel()} entries for {n} vertices'
    return dict(triangles_points=xyz.contiguous(), _triangle_indices=faces.contiguous(), vertex_weight=vw.reshape(n, 1).contiguous(), features_dc=dc.contiguous()), n


def rest_of(w, n):
    from appearance import expand_wire
    r = T(expand_wire(w)).float()
    if r.ndim == 2 and r.shape[1] == 45: r = r.reshape(-1, 15, 3)
    if r.ndim == 3 and r.shape[1] == 3 and r.shape[2] == 15: r = r.permute(0, 2, 1)
    assert r.shape == (n, 15, 3), f'expand_wire returned {tuple(r.shape)}, expected ({n}, 15, 3)'
    return r.contiguous()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage2', required=True); ap.add_argument('--deliver-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = Path(a.stage2).resolve()
    for d in (pkg / 'renderer', pkg, pkg / 'codec'): sys.path.insert(0, str(d))
    D = Path(a.deliver_dir).resolve(); out = Path(a.out)
    assets = sorted((D / 'delivery').glob('*.mslb')) if (D / 'delivery').exists() else []
    print(f'[decode] {VERSION} | deliver dir {D} | assets {[p.name for p in assets]}', flush=True)
    try:
        if not assets: raise FileNotFoundError('no *.mslb under ' + str(D / 'delivery'))
        from topology import read_asset
        cache = D / 'delivery' / 'native' if (D / 'delivery' / 'native').exists() else Path(tempfile.mkdtemp()) / 'native'
        tmp = Path(tempfile.mkdtemp())
        dec = read_asset(str(assets[0]), cache, tmp)
        print('[decode] decoded keys:', {k: (tuple(np.asarray(v).shape) if hasattr(v, '__len__') and not isinstance(v, (str, bytes)) else type(v).__name__) for k, v in dec.items()}, flush=True)
        ck, n = to_ckpt(dec); ck['features_rest'] = rest_of(dec, n); tag = 'decoded'
    except Exception as e:
        print('[decode] WARNING: decoding the asset failed (' + repr(e) + '); falling back to shared-final.pt (pre-quantisation fitted mesh)', flush=True); traceback.print_exc()
        w = torch.load(D / 'shared-final.pt', map_location='cpu', weights_only=False)
        ck, n = to_ckpt(w); ck['features_rest'] = rest_of(w, n); tag = 'prequant'; out = out.with_name(out.stem + '-prequant.pt')
    for k, v in ck.items(): assert torch.isfinite(v.float()).all(), f'{k} has non-finite values'
    torch.save(ck, out)
    print(f'[decode] {tag}: wrote {out} | vertices {n} faces {ck["_triangle_indices"].shape[0]} | ' + ' '.join(f'{k} {tuple(v.shape)} {v.dtype}' for k, v in ck.items()), flush=True)
    print('DECODE OK ' + tag)


if __name__ == '__main__':
    main()
PYEOF_DEC1

cat > $AN/render/render_ckpt_v1.py <<'PYEOF_REN1'
"""render_ckpt_v1.py (VERSION render_ckpt_v1 2026-11-10): render chosen TEST views of several checkpoints side by side (render_views_v1 + the checkpoint
loading of eval_ckpt_v1).  Cameras come from upstream's Scene(load_iteration=30000) of --model (any model dir of the scene); each --ckpt NAME=PATH is a
state dict with triangles_points, _triangle_indices, vertex_weight, features_dc, features_rest (an FP32 original, or a decoded asset from
decode_asset_v1.py).  Black background, supersample as the metrics (4).  Writes per view and checkpoint: view-<cam>-<name>.png, view-<cam>-<name>-err.png,
view-<cam>-gt.png; montage.png (one row per view: GT | ckpt1 | ckpt2 ...; the error row beneath) and psnr.txt.
  python render_ckpt_v1.py --lab $LAB --dataset $DATA --images images_2 --model /scratch/zr2245/room-indoor/original --views 2 10 \
     --ckpt FP32=/.../point_cloud_state_dict.pt --ckpt 20MB=/.../decoded.pt ... --out /scratch/zr2245/renders-b23/room
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'render_ckpt_v1 2026-11-10'


def to_u8(x): return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True); ap.add_argument('--dataset', required=True); ap.add_argument('--images', default='images_2'); ap.add_argument('--model', required=True)
    ap.add_argument('--ckpt', action='append', required=True, help='NAME=PATH, repeatable, in display order'); ap.add_argument('--views', nargs='+', type=int, default=[2, 10])
    ap.add_argument('--supersample', type=int, default=4); ap.add_argument('--err_gain', type=float, default=4.0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required')
    up = Path(a.lab).resolve() / 'upstream'; sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image, ImageDraw
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images, resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=30000, shuffle=False)
    cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid)); dev = 'cuda'
    names = [c.split('=', 1)[0] for c in a.ckpt]; paths = [c.split('=', 1)[1] for c in a.ckpt]
    print(f'[render] {VERSION} | {len(cams)} test views, rendering {a.views} | ss {a.supersample} | columns GT + {names}', flush=True)
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    views = [i for i in a.views if i < len(cams)]
    gts = {}; renders = {n: {} for n in names}; errs = {n: {} for n in names}; lines = []
    for n, p in zip(names, paths):
        w = torch.load(p, map_location='cpu', weights_only=False); nv = len(w['triangles_points'])
        mesh.vertices = w['triangles_points'].to(dev).float().reshape(nv, 3).contiguous()
        mesh._triangle_indices = w['_triangle_indices'].to(dev).to(torch.int32).reshape(-1, 3).contiguous()
        mesh.vertex_weight = w['vertex_weight'].to(dev).float().reshape(-1, 1).contiguous()
        mesh._features_dc = w['features_dc'].to(dev).float().reshape(nv, 1, 3).contiguous()
        mesh._features_rest = w['features_rest'].to(dev).float().reshape(nv, 15, 3).contiguous()
        nf = len(mesh._triangle_indices)
        for k in ('image_size', 'importance_score', 'pixel_count'): setattr(mesh, k, torch.zeros(nf, device=dev))
        mesh.scaling = a.supersample; del w
        print(f'[render] {n}: {nv} vertices {nf} faces', flush=True)
        with torch.no_grad():
            for i in views:
                c = cams[i]; im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
                psnr = float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12)))
                err = ((im - gt).abs().sum(0, keepdim=True) * a.err_gain / 3).clamp(0, 1).repeat(3, 1, 1)
                renders[n][i] = to_u8(im); errs[n][i] = to_u8(err); gts[i] = to_u8(gt)
                Image.fromarray(renders[n][i]).save(out / f'view-{c.image_name}-{n}.png'); Image.fromarray(errs[n][i]).save(out / f'view-{c.image_name}-{n}-err.png')
                lines.append(f'{c.image_name} {n} psnr {psnr:.2f}'); print(f'[render] {c.image_name} {n}: PSNR {psnr:.2f}', flush=True)
        torch.cuda.empty_cache()
    rows = []
    for i in views:
        c = cams[i]; Image.fromarray(gts[i]).save(out / f'view-{c.image_name}-gt.png')
        top = [gts[i]] + [renders[n][i] for n in names]; bot = [np.zeros_like(gts[i])] + [errs[n][i] for n in names]
        rows.append(np.concatenate(top, axis=1)); rows.append(np.concatenate(bot, axis=1))
    mont = Image.fromarray(np.concatenate(rows, axis=0)); d = ImageDraw.Draw(mont); W = gts[views[0]].shape[1]
    for j, n in enumerate(['GT'] + names): d.text((j * W + 8, 8), n, fill=(255, 255, 0))
    mont.save(out / 'montage.png'); (out / 'psnr.txt').write_text('\n'.join(lines) + '\n')
    print(f'[render] wrote {out}/montage.png ({len(views)} views x [GT | {" | ".join(names)}], error rows beneath) and per-view PNGs', flush=True)
    print('RENDER OK')


if __name__ == '__main__':
    main()
PYEOF_REN1

cat > $AN/render-b23.sbatch <<'SBEOF_REND23'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rend23
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/zr2245/analysis/rend23-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rend23-%x-%j.err
# render-b23.sbatch (2026-11-10, VERSION rend23-v1): qualitative figure for one scene.  Each entry of ASSETS is NAME=<deliver dir> (decoded with the
# stage-2 decoder by decode_asset_v1.py, inside the stage-2 env) or NAME=<state_dict.pt> (an FP32 checkpoint, used as is).  Then render_ckpt_v1.py
# renders the chosen test VIEWS of every entry next to the ground truth at the metrics' supersampling (4x), writes per-view PNGs, an error row and
# montage.png under $OUT.  A failed decode falls back to the pre-quantisation shared mesh (file name *-prequant.pt; the montage column is then
# labelled NAME~).  Nothing is modified in the source run directories.
#   sbatch --export=ALL,SCENE=room,DATA=/scratch/zr2245/datasets/m360/room,IMAGES=images_2,MODEL=/scratch/zr2245/room-indoor/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/room,ASSETS="FP32=/scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|20MB=/scratch/zr2245/room-adapt9x-t20mb/axisB/v1.0/deliver|..." --job-name=room-b23-rend .../render-b23.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=6
SCENE=${SCENE:?}; DATA=${DATA:?}; IMAGES=${IMAGES:?}; MODEL=${MODEL:?}; VIEWS=${VIEWS:-"2 10"}; OUT=${OUT:?}; ASSETS=${ASSETS:?set ASSETS="NAME=path|NAME=path" via --export}; SS=${SS:-4}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; R=$AN/render
echo "render-b23 | scene $SCENE | data $DATA/$IMAGES | cameras from $MODEL | views $VIEWS | ss $SS | out $OUT | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $R/decode_asset_v1.py $R/render_ckpt_v1.py $PKG/stage2_run.sh $MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt $LAB/upstream/train.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
[ -e $DATA/$IMAGES/. ] || [ -f $DATA/transforms_test.json ] || { echo "FAILED: dataset $DATA has neither $IMAGES/ nor transforms_test.json"; exit 1; }
mkdir -p $OUT/decoded; CK=""; nf=0
IFS='|' read -ra ENTRIES <<< "$ASSETS"
for e in "${ENTRIES[@]}"; do
  n=${e%%=*}; p=${e#*=}
  if [ -f "$p" ]; then CK="$CK --ckpt $n=$p"; echo "asset $n: checkpoint $p"; continue; fi
  if [ -d "$p" ]; then
    echo "=== decode $n from $p ($(date +%H:%M:%S)) ==="
    bash $PKG/stage2_run.sh $R/decode_asset_v1.py --deliver-dir $p --out $OUT/decoded/$n.pt > $OUT/decoded/$n.log 2>&1; grep -h "^\[decode\]\|DECODE OK\|Error\|Traceback" $OUT/decoded/$n.log | tail -6
    if [ -f $OUT/decoded/$n.pt ]; then CK="$CK --ckpt $n=$OUT/decoded/$n.pt"
    elif [ -f $OUT/decoded/$n-prequant.pt ]; then CK="$CK --ckpt $n~=$OUT/decoded/$n-prequant.pt"; echo "note: $n rendered from the pre-quantisation mesh"
    else echo "FAILED: decode $n"; nf=$((nf+1)); fi
  else echo "FAILED: $n: $p is neither a file nor a directory"; nf=$((nf+1)); fi
done
[ -n "$CK" ] || { echo "FAILED: nothing to render"; exit 1; }
echo "=== render ($(date +%H:%M:%S)) ==="
python $R/render_ckpt_v1.py --lab $LAB --dataset $DATA --images $IMAGES --model $MODEL --views $VIEWS --supersample $SS --out $OUT $CK > $OUT/render.log 2>&1 || { echo "FAILED: RENDER"; grep -n -m 8 -i 'error\|Traceback' $OUT/render.log; tail -12 $OUT/render.log; exit 1; }
grep -h "^\[render\]" $OUT/render.log; cat $OUT/psnr.txt; ls -la $OUT/montage.png
echo "render-b23 $SCENE exit 0 ($nf decode failures)  $(date)"
SBEOF_REND23

cat > $AN/submit-batch23-v1.sh <<'SHEOF_B23'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; mkdir -p $AN/render /scratch/zr2245/renders-b23
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/render/decode_asset_v1.py 67 49735f842b1cd7ef7745cbd315dcef57
chk $AN/render/render_ckpt_v1.py 75 dca70c1012c838dc025555149c8e11a1
chk $AN/render-b23.sbatch 45 2dfd6e2174218ffeed25ef9aab1d8df9
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python -m py_compile $AN/render/decode_asset_v1.py $AN/render/render_ckpt_v1.py || { echo "CHECK FAILED: py_compile"; exit 1; }; bash -n $AN/render-b23.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
miss=0; for p in /scratch/zr2245/drjohnson-adapt9x-t10mb/axisB/v1.0/deliver /scratch/zr2245/drjohnson-adapt9x-t20mb/axisB/v1.0/deliver /scratch/zr2245/drjohnson-adapt9x-t5mb/axisB/v1.0/deliver /scratch/zr2245/drjohnson-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt /scratch/zr2245/garden-adapt9x-t10mb/axisB/v1.0/deliver /scratch/zr2245/garden-adapt9x-t20mb/axisB/v1.0/deliver /scratch/zr2245/garden-adapt9x-t5mb/axisB/v1.0/deliver /scratch/zr2245/garden-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt /scratch/zr2245/lego-adapt9xb-t1.5mb-best/axisB/v1.0/deliver /scratch/zr2245/lego-adapt9xb-t3mb-best/axisB/v1.0/deliver /scratch/zr2245/lego-adapt9xb-t6mb-best/axisB/v1.0/deliver /scratch/zr2245/lego-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt /scratch/zr2245/room-adapt9x-t10mb/axisB/v1.0/deliver /scratch/zr2245/room-adapt9x-t20mb/axisB/v1.0/deliver /scratch/zr2245/room-adapt9x-t5mb/axisB/v1.0/deliver /scratch/zr2245/room-budget-486000/axisB/v1.0/deliver /scratch/zr2245/room-indoor/axisB/g0ctrl-486000/deliver /scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt /scratch/zr2245/truck-adapt9x-t10mb/axisB/v1.0/deliver /scratch/zr2245/truck-adapt9x-t20mb/axisB/v1.0/deliver /scratch/zr2245/truck-adapt9x-t5mb/axisB/v1.0/deliver /scratch/zr2245/truck-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; do [ -e $p ] || { echo "WARNING: missing $p (that column is skipped by the job)"; miss=$((miss+1)); }; done; echo "$miss missing inputs"
busy=$(squeue -u $USER -h -o %j | grep -c b23); [ "$busy" = 0 ] || { echo "CHECK FAILED: b23 jobs already queued"; exit 1; }; echo "CHECK OK"
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j $(echo "$@" | grep -o 'job-name=[^ ]*' | cut -d= -f2)" || echo "FAILED: sbatch $*"; }
sb --export=ALL,SCENE=room,DATA=/scratch/zr2245/datasets/m360/room,IMAGES=images_2,MODEL=/scratch/zr2245/room-indoor/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/room,ASSETS="FP32=/scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|20MB=/scratch/zr2245/room-adapt9x-t20mb/axisB/v1.0/deliver|10MB=/scratch/zr2245/room-adapt9x-t10mb/axisB/v1.0/deliver|5MB=/scratch/zr2245/room-adapt9x-t5mb/axisB/v1.0/deliver" --job-name=room-b23-rend $AN/render-b23.sbatch
sb --export=ALL,SCENE=room-g0,DATA=/scratch/zr2245/datasets/m360/room,IMAGES=images_2,MODEL=/scratch/zr2245/room-indoor/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/room-g0,ASSETS="FP32=/scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|grown-5MB=/scratch/zr2245/room-adapt9x-t5mb/axisB/v1.0/deliver|cap-486k=/scratch/zr2245/room-budget-486000/axisB/v1.0/deliver|collapse-486k=/scratch/zr2245/room-indoor/axisB/g0ctrl-486000/deliver" --job-name=room-g0-b23-rend $AN/render-b23.sbatch
sb --export=ALL,SCENE=garden,DATA=/scratch/zr2245/datasets/m360/garden,IMAGES=images_4,MODEL=/scratch/zr2245/garden-vis-collapse/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/garden,ASSETS="FP32=/scratch/zr2245/garden-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|20MB=/scratch/zr2245/garden-adapt9x-t20mb/axisB/v1.0/deliver|10MB=/scratch/zr2245/garden-adapt9x-t10mb/axisB/v1.0/deliver|5MB=/scratch/zr2245/garden-adapt9x-t5mb/axisB/v1.0/deliver" --job-name=garden-b23-rend $AN/render-b23.sbatch
sb --export=ALL,SCENE=truck,DATA=/scratch/zr2245/datasets/tandt/truck,IMAGES=images,MODEL=/scratch/zr2245/truck-vis-collapse/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/truck,ASSETS="FP32=/scratch/zr2245/truck-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|20MB=/scratch/zr2245/truck-adapt9x-t20mb/axisB/v1.0/deliver|10MB=/scratch/zr2245/truck-adapt9x-t10mb/axisB/v1.0/deliver|5MB=/scratch/zr2245/truck-adapt9x-t5mb/axisB/v1.0/deliver" --job-name=truck-b23-rend $AN/render-b23.sbatch
sb --export=ALL,SCENE=drjohnson,DATA=/scratch/zr2245/datasets/db/drjohnson,IMAGES=images,MODEL=/scratch/zr2245/drjohnson-indoor/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/drjohnson,ASSETS="FP32=/scratch/zr2245/drjohnson-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|20MB=/scratch/zr2245/drjohnson-adapt9x-t20mb/axisB/v1.0/deliver|10MB=/scratch/zr2245/drjohnson-adapt9x-t10mb/axisB/v1.0/deliver|5MB=/scratch/zr2245/drjohnson-adapt9x-t5mb/axisB/v1.0/deliver" --job-name=drjohnson-b23-rend $AN/render-b23.sbatch
sb --export=ALL,SCENE=lego,DATA=/scratch/zr2245/datasets/nerf_synthetic/lego,IMAGES=train,MODEL=/scratch/zr2245/lego-blender4-best/original,VIEWS="0 100",OUT=/scratch/zr2245/renders-b23/lego,ASSETS="FP32=/scratch/zr2245/lego-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|6MB=/scratch/zr2245/lego-adapt9xb-t6mb-best/axisB/v1.0/deliver|3MB=/scratch/zr2245/lego-adapt9xb-t3mb-best/axisB/v1.0/deliver|1.5MB=/scratch/zr2245/lego-adapt9xb-t1.5mb-best/axisB/v1.0/deliver" --job-name=lego-b23-rend $AN/render-b23.sbatch
sleep 20; squeue -u $USER -o "%.10i %.20j %.2t %.8M %R" | grep -E "b23|JOBID"
SHEOF_B23
bash $AN/submit-batch23-v1.sh 2>&1 | tee $AN/submit-batch23-v1.log
```

## §2 collect (15–20 min later)

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-3hours -o JobID,JobName%20,State,Elapsed,ExitCode | grep -E "b23|JobID"
for f in $AN/rend23-*-b23-rend-*.out; do echo "--- $f"; grep -h "^\[decode\] decoded keys\|^\[decode\] decoded:\|prequant\|WARNING\|FAILED\|psnr\|montage" $f | cut -c1-240; done
ls -la /scratch/zr2245/renders-b23/*/montage.png
```

Then open `/scratch/zr2245/renders-b23/<scene>/montage.png` in VS Code (right-click → Download to keep them). Send me the §2 output and, if you can, the six montage PNGs; I'll pick the views and crops for the paper figure.
