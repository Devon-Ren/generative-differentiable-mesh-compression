"""Build ONE coarse level of the original Room mesh with the pixel-space collapse criterion.  CPU, ~48 GB RAM.

Same error-adaptive endpoint QEM as the historical hierarchy (stage2/legacy_hierarchy, used for the
26.9 MB result), same appearance features (independent SH + opacity, radiance unit 0.1, 80 deg normal limit),
same legality rules.  Only the SCORE changes (native/native_qem_vis.cpp):
    old:  max( position_radius / median_edge ,  appearance_radius )                  [median-edge units]
    new:  max( position_radius * pixel_scale ,  appearance_radius * appearance_unit ) [pixels]
pixel_scale comes from vertex_footprint.py (px per world unit in the training view where the vertex is
largest), carried through clusters as the max.  Vertices no training view renders get the
--unseen-quantile quantile of the seen scales (not 0, so hidden geometry cannot grow unboundedly).
appearance_unit defaults to the median edge length in pixels, so one appearance unit still costs as much as
one median edge, exactly as in the old score.

Default mode collapses in cost order until --target-faces remain (4,041,039 = the historical error-2 level),
so the comparison with coarse-initial.pt of run room-shared-collapse-QGQgyr is at equal face count.
Experiment 2 (render sensitivity, needs footprint.npz written with sensitivity on):
    --geo-weight sens_pos   multiplies pixel_scale[v] by  clip((sens_pos[v]/median)^power, lo, hi)
    --app-weight sens_dc    multiplies appearance_unit by clip((sens_dc[v]/median)^power, lo, hi)   (per vertex)
so vertices whose displacement / colour change moves the training images most are the most expensive to merge.
--uniform reproduces the historical criterion bit-for-bit (regression check: with --limits 1,2,4 --select 2 the
face count must equal 4,041,039).

Outputs in --output: coarse-initial.pt (same keys as the Stage-2 file, so coarse_finetune.py can evaluate /
fine-tune it), selected-collapse-map.npz, report.json.

  cd /scratch/zr2245/analysis/vis_collapse && python build_vis.py      # once
  python coarse_vis.py --source /scratch/zr2245/room-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt \
      --footprint /scratch/zr2245/room-vis-collapse/footprint.npz \
      --old-map /scratch/zr2245/room-shared-collapse-QGQgyr/run/selected-collapse-map.npz \
      --output /scratch/zr2245/room-vis-collapse/pixel-target
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'native'))


def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


PACKAGE_VERSION = (Path(__file__).resolve().parent / 'VERSION').read_text().split()[0] if (Path(__file__).resolve().parent / 'VERSION').exists() else 'unknown'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def quantiles(x, qs=(.05, .25, .5, .75, .95)):
    x = np.asarray(x, np.float64)
    return {f'q{int(q * 100):02d}': float(v) for q, v in zip(qs, np.quantile(x, qs))} if x.size else {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='point_cloud_state_dict.pt of the original model')
    p.add_argument('--footprint', help='footprint.npz from vertex_footprint.py (required unless --uniform)')
    p.add_argument('--output', required=True)
    p.add_argument('--target-faces', type=int, default=4041039, help='collapse in cost order until this many faces remain (0 = use --limits)')
    p.add_argument('--target-vertices', type=int, default=0, help='v5: collapse in cost order until this many vertices remain; overrides --target-faces. '
                   'Use it to compare criteria at equal vertex count (= roughly equal bytes), since equal face count is not equal rate.')
    p.add_argument('--limits', default='', help='comma-separated increasing error limits (pixels; median-edge units with --uniform)')
    p.add_argument('--select', type=float, help='which limit to export (default: last reached)')
    p.add_argument('--uniform', action='store_true', help='historical criterion (ignore footprint)')
    p.add_argument('--appearance-unit', type=float, help='pixels per appearance unit (default: median edge length in pixels)')
    p.add_argument('--unseen-quantile', type=float, default=.05, help='pixel_scale floor for unseen vertices = this quantile of seen scales')
    p.add_argument('--scale-power', type=float, default=1., help='use pixel_scale**power (1 = pixels; 0.5 = compromise; 0 = uniform)')
    p.add_argument('--base', choices=['pixel', 'uniform'], default='uniform',
                   help='uniform (default since exp1 failed): geometric term in median-edge units like the historical score, weights multiply it; pixel: per-vertex focal/depth scale')
    p.add_argument('--weight-divide-pixel-scale', action='store_true', help='divide the weight source by pixel_scale first (removes the footprint factor from sensitivities)')
    p.add_argument('--geo-weight', choices=['none', 'sens_pos', 'l1_pos', 'residual_mean', 'max_face_pixels'], default='none')
    p.add_argument('--app-weight', choices=['none', 'sens_dc', 'sens_pos', 'residual_mean'], default='none')
    p.add_argument('--weight-power', type=float, default=1.)
    p.add_argument('--weight-clip', default='0.1,10', help='lo,hi clip of the normalised weight')
    p.add_argument('--max-angle', type=float, default=80.)
    p.add_argument('--radiance-unit', type=float, default=.1, help='historical appearance scaling (coarse_once.build)')
    p.add_argument('--old-map', help='selected-collapse-map.npz of the historical run, for overlap statistics')
    a = p.parse_args()
    import _native_qem_vis as native
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    log('vis_collapse package', PACKAGE_VERSION); save_json(out / 'config.json', dict(vars(a), package_version=PACKAGE_VERSION))
    limits = [float(x) for x in a.limits.split(',') if x]
    if a.target_vertices > 0:
        if a.target_faces > 0: log('NOTE --target-vertices given: --target-faces ignored')
        a.target_faces = 0; limits = []
    if a.target_faces <= 0 and a.target_vertices <= 0 and not limits: p.error('need --target-faces, --target-vertices or --limits')
    if a.target_faces > 0 and limits: log('NOTE --target-faces given: --limits ignored'); limits = []

    log('loading', a.source)
    s = torch.load(a.source, map_location='cpu', weights_only=False)
    keys = ('triangles_points', 'features_dc', 'features_rest', '_triangle_indices', 'vertex_weight', 'sigma', 'active_sh_degree')
    wire = {k: (s[k].detach().cpu().clone() if isinstance(s[k], torch.Tensor) else s[k]) for k in keys}
    del s
    xyz = wire['triangles_points'].numpy().astype(np.float64); faces = wire['_triangle_indices'].numpy().astype(np.int64)
    V, F = len(xyz), len(faces); log(f'{V} vertices, {F} faces')
    floor = .999
    sh = np.concatenate([wire['features_dc'].numpy(), wire['features_rest'].numpy()], 1)
    weights = floor + (1 - floor) * torch.sigmoid(wire['vertex_weight']).numpy()
    features = np.concatenate([sh.reshape(V, -1) * np.sqrt(sh.shape[1] / (4 * np.pi)) / a.radiance_unit, weights.reshape(-1, 1) / a.radiance_unit], 1).astype(np.float32)
    del sh

    pixel_scale = None; appearance_scale = None; appearance_unit = 1.; scale_info = {}
    lo, hi = [float(x) for x in a.weight_clip.split(',')]

    def weight(fp, name, seen):
        if name == 'none': return None
        if name not in fp.files: p.error(f'footprint has no {name!r}; rerun vertex_footprint.py with sensitivity on')
        w = fp[name].astype(np.float64)
        if a.weight_divide_pixel_scale: w = np.where(fp['pixel_scale'] > 0, w / np.maximum(fp['pixel_scale'], 1e-30), 0.)
        ref = float(np.median(w[seen & (w > 0)])) if (seen & (w > 0)).any() else 1.
        w = np.clip((np.maximum(w, 0) / max(ref, 1e-30)) ** a.weight_power, lo, hi)
        w[~seen] = lo                                          # unseen: cheapest allowed
        return w, dict(source=name, median_raw=ref, power=a.weight_power, clip=[lo, hi], quantiles=quantiles(w[seen]), at_lo=float((w[seen] <= lo).mean()), at_hi=float((w[seen] >= hi).mean()))
    if not a.uniform:
        if not a.footprint: p.error('--footprint required (or --uniform)')
        fp = np.load(a.footprint)
        ps = fp['pixel_scale'].astype(np.float64); seen = fp['views_seen'] > 0
        if len(ps) != V: p.error('footprint vertex count differs from checkpoint')
        floor_ps = float(np.quantile(ps[seen], a.unseen_quantile))
        ps = np.maximum(ps, floor_ps)                       # unseen -> floor; seen never below floor either
        if a.scale_power != 1.: ps = ps ** a.scale_power
        if a.base == 'uniform': ps = np.full(V, 1. / float(np.quantile(ps, .5)))   # placeholder, replaced by 1/median_edge_px below
        # median edge length in pixels (edges with both endpoints seen) -> appearance unit
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]); e.sort(1); e = np.unique(e, axis=0)
        elen = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1) * np.maximum(ps[e[:, 0]], ps[e[:, 1]])
        both = seen[e[:, 0]] & seen[e[:, 1]]
        edge_px = float(np.median(elen[both])) if both.any() else float(np.median(elen))
        if a.base == 'uniform':
            # historical units: position radius / median edge (world), appearance radius × 1  -> both dimensionless, weights multiply them
            e_world = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1); med_world = float(np.median(e_world))
            ps = np.full(V, 1. / med_world); appearance_unit = a.appearance_unit if a.appearance_unit else 1.
            scale_info['base'] = dict(mode='uniform', median_edge_world=med_world, note='cost unit = median edges; 28 px in exp1 == 2.0 here')
            edge_px = None                                   # pixel quantities are meaningless in uniform mode
        else:
            appearance_unit = a.appearance_unit if a.appearance_unit else edge_px; scale_info['base'] = dict(mode='pixel')
        gw = weight(fp, a.geo_weight, seen); aw = weight(fp, a.app_weight, seen)
        if gw: ps = ps * gw[0]; scale_info['geo_weight'] = gw[1]
        if aw: appearance_scale = appearance_unit * aw[0]; scale_info['app_weight'] = aw[1]
        scale_info.update(seen_fraction=float(seen.mean()), pixel_scale_floor=floor_ps, appearance_unit=appearance_unit,
                          scale_quantiles=quantiles(ps), **({} if edge_px is None else dict(median_edge_px_seen=edge_px, edge_px_quantiles=quantiles(elen[both]))))
        if gw: log('geo weight', a.geo_weight, gw[1]['quantiles'])
        if aw: log('app weight', a.app_weight, aw[1]['quantiles'])
        log('pixel scale floor', round(floor_ps, 2), 'median edge px', None if edge_px is None else round(edge_px, 3), 'appearance_unit', round(appearance_unit, 3))
        pixel_scale = ps; del e, elen
    target = a.target_faces if a.target_faces > 0 else 0

    def progress(row): log('collapse:', {k: (v if not isinstance(v, dict) else dict(v)) for k, v in row.items()})
    log('building', 'uniform (historical)' if a.uniform else 'pixel-space', 'target faces', target, 'target vertices', a.target_vertices, 'limits', limits)
    raw = native.build_error(xyz, faces, features, limits, a.max_angle, progress, pixel_scale, appearance_unit, target, appearance_scale, a.target_vertices)
    del features
    reached = [float(x) for x in raw['error_limits']]
    if a.select is not None:
        idx = max(i for i, e in enumerate(reached) if e <= a.select)
    else:
        idx = len(raw['levels']) - 1
    level = raw['levels'][idx]; ids = np.asarray(level['ids'], np.int64); sub_faces = np.asarray(level['faces'], np.int64)
    log(f'selected level {idx}: {len(ids)} vertices, {len(sub_faces)} faces, reached cost {reached[idx]:.4f}')

    coarse = dict(wire)
    t_ids = torch.from_numpy(ids)
    for k in ('triangles_points', 'features_dc', 'features_rest', 'vertex_weight'): coarse[k] = wire[k][t_ids].clone()
    coarse['_triangle_indices'] = torch.as_tensor(sub_faces, dtype=torch.int32).clone(); coarse['opacity_floor'] = floor
    torch.save(coarse, out / 'coarse-initial.pt')
    collapses = np.asarray(raw['collapses'])[:raw['offsets'][idx]]
    np.savez_compressed(out / 'selected-collapse-map.npz', ids=ids, face_ids=np.asarray(level['face_ids']), faces=sub_faces, collapses=collapses)

    removed = np.ones(V, bool); removed[ids] = False
    report = dict(package_version=PACKAGE_VERSION, mode='uniform' if a.uniform else a.base, target_faces=target, target_vertices=a.target_vertices, limits=limits, reached_error_limits=reached, selected_index=idx,
                  vertices=int(len(ids)), faces=int(len(sub_faces)), input_vertices=V, input_faces=F,
                  retained_face_ratio=len(sub_faces) / F, retained_vertex_ratio=len(ids) / V, merges=int(len(collapses)),
                  hierarchy={k: (dict(v) if isinstance(v, dict) else (float(v) if isinstance(v, (int, float)) else v)) for k, v in raw.items()
                             if k not in ('levels', 'collapses', 'offsets', 'error_limits')},
                  scale=scale_info)
    if pixel_scale is not None:
        report['pixel_scale_removed'] = quantiles(pixel_scale[removed]); report['pixel_scale_kept'] = quantiles(pixel_scale[~removed])
        fp_seen = np.load(a.footprint)['views_seen'] > 0
        report['removed_seen_fraction'] = float(fp_seen[removed].mean()); report['unseen_removed_fraction'] = float(removed[~fp_seen].mean()) if (~fp_seen).any() else None
    if a.old_map:
        old = np.load(a.old_map); old_removed = np.ones(V, bool); old_removed[old['ids']] = False
        inter = int((removed & old_removed).sum()); union = int((removed | old_removed).sum())
        report['old_map'] = dict(path=a.old_map, old_vertices=int(len(old['ids'])), old_faces=int(len(old['faces'])),
                                 removed_jaccard=inter / max(1, union), removed_overlap_fraction=inter / max(1, int(old_removed.sum())))
        if pixel_scale is not None: report['pixel_scale_removed_by_old'] = quantiles(pixel_scale[old_removed])
    save_json(out / 'report.json', report)
    log('report', {k: report[k] for k in ('vertices', 'faces', 'retained_face_ratio', 'merges')})
    if a.old_map: log('overlap with historical removed set: Jaccard', round(report['old_map']['removed_jaccard'], 4))
    log('wrote', out / 'coarse-initial.pt')


if __name__ == '__main__':
    main()
