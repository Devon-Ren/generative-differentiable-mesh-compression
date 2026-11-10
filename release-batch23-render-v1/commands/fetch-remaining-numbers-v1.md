# fetch-remaining-numbers-v1 (2026-11-10): CPU, seconds. Fills the last TBD cells of experiments-draft-v1.tex.

```bash
AN=/scratch/zr2245/analysis; O=$AN/fetch-remaining-$(date +%m%d-%H%M).txt
{ echo "===== Deep Blending price runs (pre-codec 'test' and delivered)"; cat $AN/rd-prune3i-drjohnson.txt $AN/rd-prune3i-playroom.txt 2>/dev/null | cut -c1-260
echo "===== Blender FP32 originals (val-selected)"; cat $AN/originals-blender-val4.txt $AN/originals-blender-val5.txt 2>/dev/null | cut -c1-220
echo "===== stream bytes: room 20 MB and 5 MB deliveries"; for T in 20 5; do echo "--- room $T MB"; python -c "import json,sys; r=json.load(open('/scratch/zr2245/room-adapt9x-t${T}mb/axisB/v1.0/deliver/delivery/codec-report.json')); pb=r['payload_bytes']; print('file_bytes', r['file_bytes']); [print(f'{k:28s} {v:>10d}  {100*v/r[\"file_bytes\"]:5.1f} %') for k,v in sorted(pb.items(), key=lambda kv:-kv[1])]"; done
echo "===== garden MPEG rows"; grep -h "^garden o3dgc" $AN/o3dgc-v1/*.txt $AN/generic-codecs*.txt 2>/dev/null | cut -c1-160 | sort -u
echo "===== Deep Blending level-1.0 of the originals (if any)"; grep -hE "^(drjohnson|playroom) " $AN/originals-indoor.txt 2>/dev/null | cut -c1-240; } > $O 2>&1; wc -l $O; echo "send me $O"
```
