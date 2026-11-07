# indoor-diag-v1 stage decomposition read (2026-11-05 14:20, CPU only, seconds)

Splits the remaining top-rate loss into "SH sharing" (coarse_input -> shared_fitted, validation) and "quantisation + coding" (shared_fitted -> decoded_delivery, validation) from the summary.json each delivery already wrote. Reads the level-1.0 baseline delivery and the two diag-v1 refits for room + kitchen. Paste the whole block on any node.

```bash
cat > /scratch/zr2245/analysis/indoor-diag-v1-stages.py <<'PY'
import json, pathlib, sys
rows = []
for S in ("room", "kitchen"):
    B = pathlib.Path(f"/scratch/zr2245/{S}-indoor")
    for name, p in (("K4096 R256 (baseline)", B / "axisB/v1.0/deliver/summary.json"),
                    ("K4096 R1024", B / "diag-v1/k4096_r1024/summary.json"),
                    ("K16384 R1024", B / "diag-v1/k16384_r1024/summary.json")):
        if not p.exists(): print(f"{S} | {name} | MISSING {p}"); continue
        j = json.load(open(p)); st = {r["stage"]: r["validation"]["psnr"] for r in j.get("stages", [])}
        need = ("coarse_input", "shared_initial", "shared_fitted", "decoded_delivery")
        if any(k not in st for k in need): print(f"{S} | {name} | summary.json lacks stages {sorted(set(need)-set(st))}; keys: {sorted(j)[:12]}"); continue
        ci, si, sf, dd = (st[k] for k in need)
        print(f"{S} | {name:22s} | xyz {j['xyz']} dc {j['dc']} | val coarse {ci:.3f} -> shared_init {si:.3f} -> shared_fit {sf:.3f} -> decoded {dd:.3f}"
              f" | sharing {sf-ci:+.3f} | quant+coding {dd-sf:+.3f} | test coarse {j['input_test']['psnr']:.3f} -> delivered {j['delivery_test']['psnr']:.3f} ({j['delivery_test']['psnr']-j['input_test']['psnr']:+.3f}) | {j['file_MB']:.3f} MB")
PY
python /scratch/zr2245/analysis/indoor-diag-v1-stages.py | tee /scratch/zr2245/analysis/indoor-diag-v1-stages.txt
```

Expected: 6 lines. "sharing" = loss of replacing per-vertex SH by dictionary codes (after the 400-step bank/DC fit); "quant+coding" = XYZ grid + DC q7 + const weight + codec round trip. Which of the two dominates decides the next lever: sharing -> dictionary size / residual depth (the fan); quant+coding -> XYZ q+1 re-export (seconds per point).
