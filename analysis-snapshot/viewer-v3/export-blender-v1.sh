#!/usr/bin/env bash
# export-blender-v1.sh (2026-11-04): decode the delivered lambda 6.2e-4 asset of each NeRF-Synthetic scene into a viewer .msv
# (CPU only, ~1 min/scene). Never exits the shell: prints FAILED lines and continues. Usage: bash export-blender-v1.sh [scenes...]
AN=/scratch/zr2245/analysis; VW=$AN/viewer-v2; V1=$AN/viewer-v1; PKG=$AN/vis_collapse-v12; OUT=$VW/msv; mkdir -p $OUT
LAM=${LAM:-0.00062}; SCENES=${*:-"chair drums ficus hotdog lego materials mic ship"}
LOG=$VW/export-blender-v1.log; : > $LOG
for f in $V1/export_viewer_v1.py $PKG/stage2_run.sh; do [ -f $f ] || { echo "FAILED: missing $f (install viewer-v1 §1 first)" | tee -a $LOG; exit 0; }; done
for S in $SCENES; do
  G=/scratch/zr2245/$S-rd3b4-lam$LAM-mu1.0; GB=$G-best
  A=$GB/axisB/v1.0/deliver/delivery/asset.mslb; [ -f $A ] || A=$(find $GB/axisB/v1.0/deliver -name asset.mslb 2>/dev/null | head -1)
  CAM=$G/original/cameras.json; [ -f "$CAM" ] || CAM=$GB/original/cameras.json; [ -f "$CAM" ] || CAM=""
  [ -n "$A" ] && [ -f "$A" ] || { echo "FAILED: no asset.mslb under $GB/axisB/v1.0/deliver" | tee -a $LOG; continue; }
  echo "=== $S lam $LAM ($(date +%H:%M:%S)) cameras: ${CAM:-none} ===" | tee -a $LOG
  bash $PKG/stage2_run.sh $V1/export_viewer_v1.py --asset $A ${CAM:+--cameras $CAM} --out $OUT/$S-lam$LAM.msv --label "$S lam$LAM" --native-cache $VW/native-cache > $VW/export-$S.log 2>&1 \
    || { echo "FAILED: export $S"; tail -8 $VW/export-$S.log; } | tee -a $LOG
  grep "^VIEWER" $VW/export-$S.log | tee -a $LOG
done
echo "----- failures -----"; grep "^FAILED" $LOG || echo none; ls -la $OUT/*.msv 2>/dev/null; du -sh $OUT
