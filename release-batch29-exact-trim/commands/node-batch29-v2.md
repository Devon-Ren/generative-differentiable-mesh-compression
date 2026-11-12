# node-batch29-v2 (2026-11-12): byte-targeted, fold-over-safe trimming (10 jobs, fairshare, about 8 GPU-h)

## What the first version showed (kitchen, 10 MB, live log)

- The diagnosis holds: the original rule delivers a fraction of the plan (e.g. 31,171 of 134,818 pairs; 6,417 of 17,309).
- Raising the candidate threshold helps early (39,465 instead of 31,171 pairs) and hurts late (6,050 instead of 14,547): the fold-over filter tests all selected collapses together and rejects every selected pair next to a flipped face, so asking for more pairs at once gets more of them rejected.
- Many of the cheapest edges lie in the part of the mesh that is not delivered: 214 pairs freed 0.000 MB, and the bytes per pair swing between 3 and 9.
- Predicted final size at 19,500: 1.073 x target (aim 1.25), 1.157 x (aim 1.35): better than +13.5%, not on target.

## What changes now

A round asks for bytes instead of pairs. `select_feasible` ranks only the edges that free delivered bytes, by error per freed byte, and builds the set in batches: each batch is tested for fold-overs together with everything already accepted, the offending pairs are banned, the rest is accepted, until the wanted bytes are reached (at most 15% of the vertices per round). CPU test with the real `rd_prune` code on synthetic meshes: 100.0-100.4% of the wanted bytes, no fold-over in the final set (the original rule: 76-113%).

| variant | bytes wanted per round | growth aim |
|---|---|---|
| feas | excess / rounds left | default (1.25 captured, 1.1 objects) |
| feasq | excess / min(rounds left, 3) | default |
| feas25, feasq25 | as above | 1.25 (objects) |
| feasq35 | as feasq | 1.35 |

Runs: kitchen 10 MB (feasq, feas, feasq35); hotdog 2.5 (feasq25, feas25); drjohnson 10 (feasq); lego 2.5 (feasq25); treehill 20 (feasq35); room 5 (feasq); garden 10 (feasq).

Success criteria are unchanged: within 5% of the target, PSNR at the same file size not more than 0.1 dB below the current method, one growth aim for captured scenes and objects.

## Step 0: free the queue (keeps the three running kitchen jobs and your cpu session)

```bash
scancel 22849902 22849906 22849900 22849914 22849909 22849923 22849930 22849897 22849898
```

The three kitchen jobs of the first version are past the trimming phase; their delivered size tells us how accurate the size estimate at 19,500 is, so let them finish.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/select_feasible_v1.py <<'FEAS29'
# select_feasible_v1 (2026-11-12, batch29 v2): the text between the two marker lines is inserted verbatim into the training script.
# >>> select_feasible
@torch.no_grad()
def _match_cheapest(e, s, V, passes=24):
    """Vertex-disjoint subset of the candidate edges e [C,2] (scores s [C], distinct): repeatedly take the edges that are the cheapest at both endpoints."""
    dev = e.device; C = e.shape[0]; picked = torch.zeros(C, dtype=torch.bool, device=dev); avail = torch.ones(V, dtype=torch.bool, device=dev)
    inf = torch.full_like(s, float('inf'))
    for _ in range(passes):
        live = ~picked & avail[e[:, 0]] & avail[e[:, 1]]
        if not bool(live.any()): break
        sl = torch.where(live, s, inf)
        vmin = torch.full((V,), float('inf'), dtype=s.dtype, device=dev)
        vmin.scatter_reduce_(0, e[:, 0], sl, reduce='amin'); vmin.scatter_reduce_(0, e[:, 1], sl, reduce='amin')
        idx = torch.nonzero(live & (sl == vmin[e[:, 0]]) & (sl == vmin[e[:, 1]]), as_tuple=True)[0]
        if idx.numel() == 0: break
        use = torch.zeros(V, dtype=torch.long, device=dev)
        use.index_add_(0, e[idx, 0], torch.ones_like(idx)); use.index_add_(0, e[idx, 1], torch.ones_like(idx))
        idx = idx[(use[e[idx, 0]] == 1) & (use[e[idx, 1]] == 1)]
        if idx.numel() == 0: break
        picked[idx] = True; avail[e[idx].reshape(-1)] = False
    return picked


@torch.no_grad()
def select_feasible(pos, faces, face_err, kept_v, kept_f, b_attr, b_face, want_bytes, max_pairs, max_iters=40):
    """Collapses that free `want_bytes` of the delivered file, cheapest error per freed byte first, as a set that is vertex-disjoint and
    free of fold-overs when applied together.  Freed bytes of edge (u,v): b_attr if both endpoints are in the delivered set + b_face per
    delivered face on the edge; edges that free nothing are not candidates.  The set is built in batches: each batch is matched, tested
    together with everything accepted so far, the pairs that flip a face are banned, the rest is accepted.
    Returns (edges [E,2], sel [P] edge indices, stats)."""
    dev = pos.device; f = faces.long(); V = pos.shape[0]
    edges, inv, cnt = edge_table(f); E = edges.shape[0]; err = face_err.double().to(dev)
    D_v = torch.zeros(V, dtype=torch.float64, device=dev)
    for k in range(3): D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(E, dtype=torch.float64, device=dev); D_uv.index_add_(0, inv, err.repeat(3))
    kv = kept_v.to(dev).bool(); kf = kept_f.to(dev).bool()
    cnt_k = torch.zeros(E, dtype=torch.float64, device=dev); cnt_k.index_add_(0, inv, kf.double().repeat(3))
    dR = float(b_attr) * (kv[edges[:, 0]] & kv[edges[:, 1]]).double() + float(b_face) * cnt_k
    priced = dR > 0
    score = torch.where(priced, (D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv) / torch.where(priced, dR, torch.ones_like(dR)), torch.full_like(dR, float('inf')))
    gen = torch.Generator(device=dev); gen.manual_seed(4321 + E)
    fin = score[priced]; scale = float(fin.mean()) if fin.numel() else 1.0
    s = score + torch.rand(E, dtype=torch.float64, device=dev, generator=gen) * 1e-9 * max(scale, 1e-30)  # distinct scores (ties: faces with no error)
    order = torch.argsort(s); order = order[priced[order]]
    n0 = _face_normals(pos, f)
    avail = torch.ones(V, dtype=torch.bool, device=dev); banned = torch.zeros(E, dtype=torch.bool, device=dev)
    pos2 = pos.clone(); new_id = torch.arange(V, device=dev)
    acc = []; n_acc = 0; acc_bytes = 0.0; n_banned = 0; it = 0; n_dropped_batches = 0
    while it < max_iters and acc_bytes < want_bytes and n_acc < max_pairs:
        it += 1
        cand = order[~banned[order] & avail[edges[order, 0]] & avail[edges[order, 1]]]
        if cand.numel() == 0: break
        mean_b = (acc_bytes / n_acc) if n_acc else float(dR[cand[:2000]].mean())
        need = max(1, min(max_pairs - n_acc, int(0.7 * (want_bytes - acc_bytes) / max(mean_b, 1e-6)) + 1))  # 70 % of what is left per batch: the last batches are small, so the total ends within a few pairs of the wanted bytes
        cand = cand[:4 * need + 256]
        new = cand[_match_cheapest(edges[cand], s[cand], V)][:need]
        converged = False
        for _ in range(10):
            if new.numel() == 0: break
            u, v = edges[new, 0], edges[new, 1]
            p2 = pos2.clone(); mid = 0.5 * (pos[u] + pos[v]); p2[u] = mid; p2[v] = mid
            nid = new_id.clone(); nid[v] = u; f2 = nid[f]
            degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
            flipped = ((n0 * _face_normals(p2, f)).sum(1) < 0) & ~degenerate
            if not bool(flipped.any()): converged = True; break
            pair_of_v = torch.full((V,), -1, dtype=torch.long, device=dev); kk = torch.arange(new.numel(), device=dev); pair_of_v[u] = kk; pair_of_v[v] = kk
            bad = pair_of_v[f[flipped].reshape(-1)]; bad = torch.unique(bad[bad >= 0])
            if bad.numel() == 0: break
            keep = torch.ones(new.numel(), dtype=torch.bool, device=dev); keep[bad] = False
            banned[new[~keep]] = True; n_banned += int(bad.numel()); new = new[keep]
        if not converged or new.numel() == 0:
            if new.numel(): banned[new] = True; n_dropped_batches += 1
            continue
        u, v = edges[new, 0], edges[new, 1]; mid = 0.5 * (pos[u] + pos[v]); pos2[u] = mid; pos2[v] = mid; new_id[v] = u
        avail[u] = False; avail[v] = False
        acc.append(new); n_acc += int(new.numel()); acc_bytes += float(dR[new].sum())
    sel = torch.cat(acc) if acc else torch.zeros(0, dtype=torch.long, device=dev)
    stats = dict(n_pairs=int(sel.numel()), planned_bytes=acc_bytes, want_bytes=float(want_bytes), n_banned=n_banned, iters=it, n_priced=int(priced.sum()), n_edges=E, dropped_batches=n_dropped_batches,
                 score_max=(float(score[sel].max()) if sel.numel() else None))
    return edges, sel, stats
# <<< select_feasible
FEAS29
cat > $AN/make_batch29_files_v2.py <<'PYGEN29B'
#!/usr/bin/env python3
"""make_batch29_files_v2.py (2026-11-12): exact trimming, second version.
What the first version showed (kitchen, 10 MB): raising the candidate threshold helps in the first rounds (39k pairs instead of 31k) and
hurts in the last ones (6k instead of 14k).  The fold-over filter tests all selected collapses together and rejects every selected pair
that touches a flipped face, so the more pairs are asked for at once, the more are rejected.  Many of the cheapest edges are also in the
part of the mesh that is not delivered and free no bytes.
Change (--trim_exact 3 or 4): the round asks for BYTES, not pairs.  select_feasible() ranks the edges that free delivered bytes by error per
freed byte and builds the set in batches; each batch is tested for fold-overs together with everything accepted so far, the offending pairs
are banned and the rest accepted, until the wanted bytes are reached (at most prune_max_frac of the vertices per round).
  3 : bytes wanted = excess / rounds left          4 : bytes wanted = excess / min(rounds left, 3)   (reacts faster to a late rise of the estimate)
Outputs train_rd_adapt_v9b_abl4.py and ablation-b24-r5{,-blender}.sbatch with ABL = feas | feasq (default growth aim), feas25 | feasq25 (aim 1.25), feasq35 (aim 1.35).
  python3 make_batch29_files_v2.py --train <train_rd_adapt_v9b_abl3.py> --sbatch <ablation-b24-r4.sbatch> --sbatch_blender <ablation-b24-r4-blender.sbatch> --feasible <select_feasible_v1.py> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': 'a30a25faa3c204d0551604ce78255122', 'sbatch': 'd16aa0d2f6fd254fa0b680c1f0c9043a', 'blender': '79f9ede7b9b0207563c471b486fd965c', 'feasible': '76bcd82359d901c2a8ad0111c3dfe6da'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--feasible', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
fs = pathlib.Path(a.feasible).read_text()
if md5(fs) != IN['feasible']: sys.exit(f'FAILED: {a.feasible} md5 {md5(fs)} != {IN["feasible"]}')
block = fs[fs.index('# >>> select_feasible'):fs.index('# <<< select_feasible')] + '# <<< select_feasible\n\n\n'
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "from rd_prune import select_collapses, collapse_edges, edge_table  # noqa: E402", "from rd_prune import select_collapses, collapse_edges, edge_table, _face_normals  # noqa: E402", 'import')
s = rep(s, "def rank_threshold(score, n_wanted, margin=4.0):", block + "def rank_threshold(score, n_wanted, margin=4.0):", 'select_feasible block')
s = rep(s, "                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False",
        "                if te >= 3: wanted = (max(1.0, min(desired, excess_pairs)) if excess > 0 else 0.0); cap_bound = False  # batch29 v2: any excess opens the round\n"
        "                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False", 'wanted v2')
s = rep(s, "                    if te:  # batch29: deliver the wanted pairs", "                    if te in (1, 2):  # batch29: deliver the wanted pairs", 'old loop gate')
s = rep(s, "                    lam_sp = shadow_price(edges, score, pairs, rd['prune_mu'])\n",
        "                    if te >= 3:  # batch29 v2: byte-targeted, fold-over-safe selection on the delivered set\n"
        "                        first_pairs = int(ps['n_pairs']); want_bytes = excess / (rounds_left if te == 3 else min(rounds_left, 3)); cap_pairs = int(rd['prune_max_frac'] * V0)\n"
        "                        edges_f, sel_f, st_f = select_feasible(triangles.vertices.detach(), triangles._triangle_indices, dD, kept_v, kept_f, size.b_v, size.b_f, want_bytes, cap_pairs)\n"
        "                        if sel_f.numel() > 0:\n"
        "                            pairs = edges_f[sel_f]; ps['n_pairs'] = int(pairs.shape[0]); ps['n_flip_rejected'] = st_f['n_banned']; ps['max_pairs'] = cap_pairs; count_driven = True\n"
        "                        ev.update(trim_exact=te, trim_want_mb=round(want_bytes / 1e6, 4), trim_planned_mb=round(st_f['planned_bytes'] / 1e6, 4), trim_batches=st_f['iters'], trim_banned=st_f['n_banned'],\n"
        "                                  trim_first_pairs=first_pairs, trim_goal=int(st_f['n_pairs']), trim_priced=st_f['n_priced'], trim_dropped_batches=st_f['dropped_batches'])\n"
        "                        print(f\"[RD prune {iteration}] feasible trim (mode {te}): wanted {want_bytes / 1e6:.3f} MB, selected {st_f['n_pairs']} pairs freeing {st_f['planned_bytes'] / 1e6:.3f} MB in {st_f['iters']} batches ({st_f['n_banned']} pairs banned by the fold-over test, {st_f['n_priced']} of {st_f['n_edges']} edges free bytes); the original rule gave {first_pairs} pairs\")\n"
        "                    lam_sp = shadow_price(edges, score, pairs, rd['prune_mu'])\n", 'feasible call')
s = rep(s, "parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2],", "parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2, 3, 4],", 'choices')
(out / 'train_rd_adapt_v9b_abl4.py').write_text(s); print(f'train_rd_adapt_v9b_abl4.py: {len(s.splitlines())} lines md5 {md5(s)}')
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r5.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r5-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "exact|exact25|exact35|exactfl via --export}", "exact|exact25|exact35|exactfl|feas|feasq|feas25|feasq25|feasq35 via --export}", 'ABL prompt')
    s = rep(s, "  exactfl) XARGS=\"--trim_exact 2\" ;;\n", "  exactfl) XARGS=\"--trim_exact 2\" ;;\n  feas) XARGS=\"--trim_exact 3\" ;;\n  feasq) XARGS=\"--trim_exact 4\" ;;\n  feas25) XARGS=\"--trim_exact 3\"; OVER=1.25 ;;\n  feasq25) XARGS=\"--trim_exact 4\"; OVER=1.25 ;;\n  feasq35) XARGS=\"--trim_exact 4\"; OVER=1.35 ;;\n", 'feas cases')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl3.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl4.py;", 'TR')
    s = s.replace("VERSION abl24-r4", "VERSION abl24-r5").replace("VERSION abl24b-r4", "VERSION abl24b-r5")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
PYGEN29B
python3 $AN/make_batch29_files_v2.py --train $AN/rdgrowth/train_rd_adapt_v9b_abl3.py --sbatch $AN/ablation-b24-r4.sbatch --sbatch_blender $AN/ablation-b24-r4-blender.sbatch --feasible $AN/select_feasible_v1.py --out $AN/batch29-build2 && cp $AN/batch29-build2/train_rd_adapt_v9b_abl4.py $AN/rdgrowth/ && cp $AN/batch29-build2/ablation-b24-r5.sbatch $AN/batch29-build2/ablation-b24-r5-blender.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch29-v2.sh <<'SUB29B'
#!/usr/bin/env bash
# submit-batch29-v2.sh (2026-11-12): pilot of the byte-targeted, fold-over-safe trimming (10 jobs, public partition, about 8 GPU-h).
#   feas    : bytes wanted per round = excess / rounds left            feasq : excess / min(rounds left, 3)
#   feas25 / feasq25 : same with growth aim 1.25 (objects)             feasq35 : feasq with growth aim 1.35
# Reference (current method): kitchen 10 MB +13.5%, drjohnson 10 +12.0%, playroom 10 +8.3%, room 5 +6.1%, garden 10 +2.0%, treehill 20 -6.6%,
#   hotdog 2.5 +9.2% (aim 1.25: +20.6%), lego 2.5 -0.2% (aim 1.25: +11.4%).
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl4.py 1053 c36e97b1e56f17e51291bec76833c3fe
chk $AN/ablation-b24-r5.sbatch 213 0947c52ac846f35b14141dcd0efa9db8
chk $AN/ablation-b24-r5-blender.sbatch 215 792b10a43264e763ea2882bbba29a488
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl4.py && bash -n $AN/ablation-b24-r5.sbatch && bash -n $AN/ablation-b24-r5-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
grep -q "^def _face_normals" $AN/rdgrowth/rd_prune.py && grep -q "^def edge_table" $AN/rdgrowth/rd_prune.py || { echo "CHECK FAILED: $AN/rdgrowth/rd_prune.py lacks _face_normals / edge_table"; exit 1; }
echo "CHECK OK"
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j  (${*:3:1})" || echo "FAILED: sbatch $*"; }
C="SMOKE=0,SELFCHECK=0"
for A in feasq feas feasq35; do sb --time=02:30:00 --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,$C,ABL=$A,TARGETS=10 --job-name=kitchen-$A-b29 $AN/ablation-b24-r5.sbatch; done
for A in feasq25 feas25; do sb --time=02:30:00 --export=ALL,SCENE=hotdog,XYZ=q14,INDOOR=1,$C,ABL=$A,TARGETS=2.5 --job-name=hotdog-$A-b29 $AN/ablation-b24-r5-blender.sbatch; done
sb --time=02:30:00 --export=ALL,SCENE=drjohnson,INDOOR=1,IMAGES=images,XYZ=q15,$C,DATA=$D/db/drjohnson,ABL=feasq,TARGETS=10 --job-name=drjohnson-feasq-b29 $AN/ablation-b24-r5.sbatch
sb --time=02:30:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,$C,ABL=feasq25,TARGETS=2.5 --job-name=lego-feasq25-b29 $AN/ablation-b24-r5-blender.sbatch
sb --time=02:30:00 --export=ALL,SCENE=treehill,INDOOR=0,IMAGES=images_4,XYZ=q16,$C,ABL=feasq35,TARGETS=20 --job-name=treehill-feasq35-b29 $AN/ablation-b24-r5.sbatch
sb --time=02:30:00 --export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,$C,ABL=feasq,TARGETS=5 --job-name=room-feasq-b29 $AN/ablation-b24-r5.sbatch
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,$C,ABL=feasq,TARGETS=10 --job-name=garden-feasq-b29 $AN/ablation-b24-r5.sbatch
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b29|JOBID"
SUB29B
bash $AN/submit-batch29-v2.sh
```

Expected: three generator lines (`train_rd_adapt_v9b_abl4.py: 1053 lines md5 c36e97b1e56f17e51291bec76833c3fe`, `ablation-b24-r5.sbatch: 213 lines md5 0947c52ac846f35b14141dcd0efa9db8`, `ablation-b24-r5-blender.sbatch: 215 lines md5 792b10a43264e763ea2882bbba29a488`), `CHECK OK`, 10 `submitted` lines.

## Watch (rounds start at iteration 12,500)

```bash
for G in kitchen-abl24-feasq-t10mb kitchen-abl24-feas-t10mb kitchen-abl24-feasq35-t10mb hotdog-abl24b-feasq25-t2.5mb hotdog-abl24b-feas25-t2.5mb; do echo "== $G"; python3 -c "
import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e.get('event') == 'prune': print('it %5d | file now %.3f x target, predicted final %.3f x | wanted %.3f MB, selected %6d pairs freeing %.3f MB (%s batches, %s banned), old rule %6s pairs | removed %.3f MB | V %d -> %d' % (e['iteration'], e['probe_ratio'], e['probe_est_ratio'], e.get('trim_want_mb', 0), e.get('trim_goal', 0), e.get('trim_planned_mb', 0), e.get('trim_batches', '-'), e.get('trim_banned', '-'), e.get('trim_first_pairs', '-'), e.get('removed_mb', 0), e['vertices_before'], e['vertices_after']))
" /scratch/zr2245/$G/original/rd_log.jsonl 2>/dev/null; done
```

If a job stops near 12,500, the error is at the end of its `train.log`: `tail -30 /scratch/zr2245/kitchen-abl24-feasq-t10mb/train.log`.

## Collect (when `squeue --me | grep -c b29` is 0)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch29-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== rows"; grep -h "abl24-exact\|abl24-feas" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//; s/, rounds \(it, probe MB.*//'; echo; echo "===== sacct"; sacct -X --starttime=now-3days --format=JobID,JobName%28,State,Elapsed,ExitCode | grep b29; grep -l "FAILED\|Traceback" $AN/abl24*-*-b29-*.out $AN/abl24*-*-b29-*.err 2>/dev/null; } > $O 2>&1; wc -l $O; echo $O
```
