#!/usr/bin/env bash
# export-fps-v1.sh (2026-11-15): viewer files (.msv) of the hold deliveries for the frame-rate measurement.  CPU only, about 1 minute per file.
# garden, room, bicycle at 20 / 10 / 5 MB.  With FP32=1 also the FP32 model of each scene (large files).  Never exits the shell on a failure.
# Output: $AN/viewer-fps/msv/<scene>-<target>mb.msv ; copy them and viewer-v3.html to the Mac, open the html in Chrome, load a file,
# choose the dataset resolution, press "benchmark (300 frames)" and note the line it prints.
AN=${AN:-/scratch/zr2245/analysis}; ROOT=${ROOT:-/scratch/zr2245}; PKG=$AN/vis_collapse-v12; V1=$AN/viewer-v1; OUT=$AN/viewer-fps/msv; mkdir -p $OUT
SCENES=${SCENES:-"garden room bicycle"}; TARGETS=${TARGETS:-"20 10 5"}
for f in $V1/export_viewer_v1.py $PKG/stage2_run.sh; do [ -f $f ] || { echo "FAILED: missing $f"; exit 0; }; done
for S in $SCENES; do
  for T in $TARGETS; do G=$ROOT/$S-abl24-hold-t${T}mb
    A=$(find $G -name asset.mslb 2>/dev/null | head -1); CAM=$(find $G -maxdepth 3 -name cameras.json 2>/dev/null | head -1)
    [ -n "$A" ] || { echo "FAILED: no asset.mslb under $G"; continue; }
    bash $PKG/stage2_run.sh $V1/export_viewer_v1.py --asset $A ${CAM:+--cameras $CAM} --out $OUT/$S-${T}mb.msv --label "$S ${T}MB" > $OUT/$S-${T}mb.log 2>&1 && echo "ok $OUT/$S-${T}mb.msv $(du -h $OUT/$S-${T}mb.msv | cut -f1)" || { echo "FAILED: export $S $T"; tail -4 $OUT/$S-${T}mb.log; }
  done
  if [ "${FP32:-0}" = 1 ]; then
    C=$(ls $ROOT/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt 2>/dev/null | head -1); CAM=$(ls $ROOT/$S-vis-collapse/original/cameras.json 2>/dev/null | head -1)
    [ -n "$C" ] || { echo "FAILED: no FP32 checkpoint for $S under $ROOT/$S-vis-collapse/original"; continue; }
    bash $PKG/stage2_run.sh $V1/export_viewer_v1.py --ckpt $C ${CAM:+--cameras $CAM} --out $OUT/$S-fp32.msv --label "$S FP32" > $OUT/$S-fp32.log 2>&1 && echo "ok $OUT/$S-fp32.msv $(du -h $OUT/$S-fp32.msv | cut -f1)" || { echo "FAILED: export $S FP32"; tail -4 $OUT/$S-fp32.log; }
  fi
done
echo "export-fps-v1 done  $(date)"
