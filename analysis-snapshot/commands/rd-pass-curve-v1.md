# rd-pass-curve-v1（2026-10-19）：从已有 rd_log.jsonl 抽每个 interval 的 n_pass / n_split / ratio 分位数（登录节点 CPU，0 GPU）

回答 C4b"有没有自然停止"：pass 数随迭代是否在下降、下降多快。覆盖 Room 6 个 λ（v1）+ 六场景 2 个 λ（v2）+ Room 剪枝 2 个（v3）。

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/rd-pass-curve-v1.txt; : > $OUT
for d in /scratch/zr2245/room-rd-error-lam* /scratch/zr2245/*-rd2-error-lam* /scratch/zr2245/room-rd3-lam*; do
  f=$d/original/rd_log.jsonl; [ -f $f ] || { echo "no log: $d" | tee -a $OUT; continue; }
  python - "$f" "$(basename $d)" <<'PY' | tee -a $OUT
import json, sys
ev=[json.loads(l) for l in open(sys.argv[1])]; c=[e for e in ev if e['event']=='config'][0]; d=[e for e in ev if e['event']=='densify']
print(sys.argv[2], 'lam', c['lam'], 'norm', c.get('dd_norm','none'), '| iter:pass/n_tri(split) ->',
      ' '.join(f"{e['iteration']}:{e['n_pass']}/{e['n_tri']}({e['n_split']})" for e in d),
      '| pass_frac first/last %.4f/%.4f' % (d[0]['n_pass']/d[0]['n_tri'], d[-1]['n_pass']/d[-1]['n_tri']),
      '| q50/q90/q99 last %.2e/%.2e/%.2e' % (d[-1]['ratio_q50'], d[-1]['ratio_q90'], d[-1]['ratio_q99']))
PY
done
echo "===== $OUT ====="; cat $OUT
```

给我 `cat /scratch/zr2245/analysis/rd-pass-curve-v1.txt` 的输出（每个 run 一行）。
