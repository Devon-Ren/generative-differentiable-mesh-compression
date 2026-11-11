# batch24 follow-up (CPU only, seconds): pre-codec test PSNR of the controller meshes, for the "same vertex count, no codec" half of Table E2

The vertex-cap rows already carry their pre-codec test PSNR (`| test X |` before `delivered`). The controller rows do not; it is in each run's
`axisB/v1.0/finetune/summary.json` (`test_best`, the same 3000-step fine-tuned FP32 mesh the cap rows report). Login node, any dir:

```bash
O=/scratch/zr2245/analysis/batch24-precodec-$(date +%Y-%m-%d-%H%M).txt
python3 - > $O <<'PY'
import json, os
def row(tag, s, t, d):
    f = d + '/axisB/v1.0/finetune/summary.json'; r = d + '/axisB/v1.0/report.json'
    if not os.path.exists(f): print(f'{tag} {s} {t} MISSING {f}'); return
    j = json.load(open(f)); b = json.load(open(r)) if os.path.exists(r) else {}
    tb = j.get('test_best') or {}; print(f"{tag} {s} {t} precodec_test {tb.get('psnr')} ssim {tb.get('ssim')} lpips {tb.get('lpips')} V {b.get('vertices')} F {b.get('faces')} | {f}")
for s in 'room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom'.split():
    for t in ('5', '7.5', '10', '15', '20'):
        d = f'/scratch/zr2245/{s}-adapt9x-t{t}mb'
        if os.path.isdir(d): row('ctrl', s, t, d)
for s in 'lego chair drums ficus hotdog materials mic ship'.split():
    for t in ('1.5', '2.5', '3', '4', '6'):
        d = f'/scratch/zr2245/{s}-adapt9xb-t{t}mb-best'
        if os.path.isdir(d): row('ctrl', s, t, d)
PY
wc -l $O; grep -c MISSING $O; echo $O
```

Send me the file. (The 7.5 / 15 MB and 3 / 6 MB rows are included when present so the appendix per-scene table can carry the same column.)
