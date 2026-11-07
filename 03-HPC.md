# 03 — HPC 全部信息（NYU Torch，用户 zr2245），截至 2026-11-07 08:00Z

Claude 不能直接连 HPC：所有信息来自 Wenhao 的粘贴和我们写的 sbatch。接手后第一件事：让 Wenhao 跑 §9 的"现场核对"块。

## 1. 集群、登录、习惯

- 集群是 **NYU Torch**（Greene 已退役）。登录 `login.torch.hpc.nyu.edu`；**永远不要写 `ssh greene`**。
- Wenhao 的日常：`torch_connect.sh`（本包 `uploads/torch_connect.sh`）申请一个小 CPU job（cpu_short，2 CPU / 8 GB / 6 h，account torch_pr_38_general），VS Code Remote-SSH 连到它（主机名 `cpu-torch`，例如 cs601）。轻 CPU 工作直接在那 `python ...`；`squeue` 里常年有一个 `cpu-torch` job（例如 22072398）。
- GPU：他**拿到一个节点后所有事都在那跑**（CPU 重活也在 GPU 节点，内存向 GPU allocation 要）；不要让他另开大内存 CPU 节点。长批次用 sbatch。
- 终端规则：粘贴块里**绝不能**有 `set -e` / 裸 `exit 1`（会杀 VS Code 终端并强制 MFA 重连）；长循环写进 heredoc 脚本再 `bash`；每块重定义变量；脚本 fail loudly，每阶段打印一行，出错打印 log tail。
- `sacct` 必须 `-u $USER`；`scontrol top` 用户无权限。

## 2. Account / QoS / reservation

- 默认 account：**`torch_pr_38_general`**（所有 sbatch）。第二个 account `torch_pr_38_tandon_advanced`（10/19 起 fairshare ~0.1，一个 job 等了 3 h；10/31 sshare：general 0.289 / RawUsage 214k，tandon 0.201 / RawUsage 0 → 大批次可分摊，例如 hull-6：6 job general + 5 tandon）。11/01 两个 account 的优先级都基本耗尽 → 申请了节点。
- **Reservation（11/05 → 11/12 09:35）**：`yw523-l40s`，节点 **gl030**（4 × L40S，`Gres=gpu:l40s:4`，128 核），Users=zr2245，Flags IGNORE_JOBS,SPEC_NODES,MAGNETIC。每个 sbatch 加 `--reservation=yw523-l40s`。节点上已有的 job 先跑完；可抢占 job 最多 30 min 内被抢占 → 新 job 可能等 ~30 min。
- 节点核心与别人共享（11/05 一个 af3 pipeline 占了 ~80 核）：GPU job 用 `--cpus-per-task=6`，否则 4 卡起不满；先看 `scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"`。QoS 限制 gpu48（16 GPU/user），squeue 显示 `QOSMaxGRESPerUser` 时多半是核心不够。可 `scontrol update JobId=X NumCPUs=6`。
- 资源默认：`--mem=32G`（Mip-360 大场景的 adapt sbatch 用 48G；coarse_vis 曾需要 ≥ 48 GB）、`--gres=gpu:1`、短 walltime（2–3 h，Blender 1.5 h，sweep 3 h）让 pending job 能 backfill；`scontrol update JobId=... TimeLimit=` 可改 PD job。
- FairShare 计 GPU-h 不计 job 数：砍 run，不砍 job。

## 3. 环境激活（每个 sbatch / 终端块开头）

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126           # Py 3.11.16, torch 2.7.1+cu126, CUDA 12.6, GsTaichi 4.6.0
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
```
- Py 3.11：f-string 里不能嵌同引号；`fused_ssim` 未装（python SSIM）；LPIPS VGG 权重在 conda env 的 lpips 包里；pip 装过 scikit-image、fast-simplification（P2b）；**open3d 不可用**（compute 节点无 libEGL）。
- Metric3D normals（Mip-360 原模型训练）需要 mmengine/mmcv-lite/timm（在 .venv），checkpoint 缓存 `/scratch/zr2245/cache/torch`。
- stage2 编码器子进程用 `STAGE2_ENV=/scratch/zr2245/conda-envs/pmsh-cu126`。

## 4. 目录（全部在 `/scratch/zr2245/`）

| 变量（sbatch 里的名字） | 路径 | 内容 |
|---|---|---|
| `LAB` | `meshsplatting-original-lossless-v1` | Wenhao 的 MeshSplatting 工作副本：`upstream/`（train.py 等，**从不改**；唯一例外：11/04 他自己改了 `dataset_readers.py:315 np.byte → np.uint8`，备份 `.bak-20261104`），`.venv` |
| `AN` | `analysis` | 我们所有工具与结果：`rdgrowth/`（train_rd_v3、rd_prune、rd_growth_v2、train_rd_adapt_v1…v7、probe_bytes_v1）、`blender/`（train_ckpt_v2、val_select_v3 …）、`diag/`、`vis_collapse-v12/`（= `PKG`）、各 sbatch（`adapt-v7-blender.sbatch` 等直接放在 `$AN/`）、提交脚本（`submit-batch8.sh`…）、结果表 `*.txt`、job 输出 `*.out/.err` |
| `PKG` | `analysis/vis_collapse-v12` | level 协议工具（vertex_footprint / coarse_vis / deliver_coarse / stage2_run.sh / scripts/run_garden.sh）；旧版 vis_collapse-vN 并存 |
| `STAGE2` | `shared-collapse-stage2-v2` | 交付 codec（stage2-v2 patch 打在 v1 的 COPY 上；`STAGE2_VERSION`）；v1 `shared-collapse-stage2-v1` 未动 |
| `VS` / `CK` | `analysis/blender/val_select_v3.py` / `analysis/blender/train_ckpt_v2.py` | Blender 协议：checkpoint 包装器 + 官方 val 选 checkpoint 并重算 importance |
| 旧路线代码 | `directional-sharing-v1`, `directional-sharing-residual-v1`, `delivery-quantization-v1`, `mesh-local-codec-v1`, `topology-order-v1`, `shared-collapse-stage2-v1` | 10/09 之前的 dense / collapse 链路（本包 legacy-packages） |
| rd-complete 包 | `meshsplatting-rd-complete-v1`（若已解压） | 10/26 一键链路包；**HPC 上的 setup.sh/run_rd.sh 从未跑过**（本地验证） |
| 数据集 | `datasets/m360/<scene>`（images_2 indoor / images_4 outdoor，sparse/0/cameras.bin），`datasets/tandt/{truck,train}`（images），`datasets/nerf_synthetic/<scene>`（transforms_{train,val,test}.json；来自 nerfbaselines HF 镜像） | |
| 缓存 | `cache/torch` | Metric3D ckpt |

### 结果 run 目录命名（`/scratch/zr2245/`）

- Mip-360 / T&T 固定 λ：`<scene>-rd3b4-lam<L>-mu1.0`（v3b sbatch；indoor 重跑 `-rd3i`…）；originals `<scene>-vis-collapse`（早期）/ 各 batch 的 originals 目录；axis 工作目录 `<scene>-vis-collapse`（`$W`，axisA-<level>/、axisB/vX/）。
- Blender：originals `<scene>-blender2`；固定 λ `<scene>-rd3b4-lam{0.00062,0.0062}-mu1.0`，val 选中副本 `…-best`（v2 交付）/ `…-best3`（val_select_v3 重交付）；adapt `<scene>-adapt<N>b-t<T>mb`（+ `-best`）；smoke `analysis/adapt<N>b-smoke-<scene>-<jobid>`。
- Mip-360 adapt：`<scene>-adapt<N>-t<T>mb`（v7b sweep = `-adapt7-`，v8 = `-adapt8-`；Blender v7b = `-adapt7b-`，v8 = `-adapt8b-`）；probe self-check mesh `<scene>-adapt1-t10mb/axisB/v1.0`（room/kitchen/garden/bicycle）、Blender `<scene>-rd3b4-lam0.0062-mu1.0-best/axisB/v1.0`。
- 每个 run：`original/`（upstream 布局：point_cloud/iteration_30000/point_cloud_state_dict.pt、chkpnt、rd_log.jsonl、rd_summary.json、probes/、train.log）、`axisB/v1.0/`（level-1.0：footprint → coarse → finetune/ → deliver/{summary.json, shared-final.pt, delivery/asset.mslb} → report.json）、`val_select.log/json`、`level.log`。

### 结果表（`$AN/*.txt`，每行一个交付点，由 sbatch 末尾的 python 块 `tee -a`）

`rd-sweep-room.txt`, `rd-sweep2-<scene>.txt`, `rd-prune3-<scene>.txt`, `rd-ctrl-room.txt`, `rd-topo-v5-room.txt`, `lambda-k-fan-v1.txt`/`v2`, `ba-sens-v1.txt`, `indoor-diag-v1.txt`, `adapt-v1.txt` … `adapt-v7.txt`, `adapt-v5-blender.txt`, `adapt-v6-blender.txt`, `adapt-v7-blender.txt`, `redeliver-v3-blender.txt`, `planarity-v1.txt`, `texrefit-v1.txt`, `p2b-lego-v2.txt`。Wenhao 把表贴回来，我们存成 `analysis/rd/<name>-<date>.txt` + csv + 分析脚本。本包 `analysis-snapshot/rd/` 有全部已贴回的表。

## 5. sbatch 约定（最新协议 = `rd/adapt-v9-blender.sbatch` / `rd/adapt-v9.sbatch`（= v8 的 + smoke 断言 `remaining`）；v7b 的两个 = 跑完 47-job sweep 的版本）

- 头：`--job-name`, `--gres=gpu:1`, `--cpus-per-task=8`（提交时覆盖成 6）, `--mem=32G/48G`, `--time`, `--output=/scratch/zr2245/analysis/<tag>-%x-%j.out`；`set -o pipefail`（不是 `set -e`）；变量全部 `${VAR:-default}` 或 `${VAR:?}` 从 `--export=ALL,SCENE=…,XYZ=qN,TARGETS="…"` 来。
- 启动自检（长步骤之前就 fail）：文件存在（`-f` 文件 / `-d` 目录 / `-L` 链接 要分清）、版本 grep（`version='adapt-v7b'`、`PROBE_VERSION = 'probe-v1'`、`VERSION = 'val_select_v3`、`VERSION = 'train_ckpt_v2`）、`py_compile`、`bash -n`、XYZ 必须 q13–q18、upstream 假设 grep（如 `triangles.scaling`）。
- 阶段：probe self-check → smoke（v7b 1300 it，强制 prune 1250；v8 1600 it，1500 强制 phase-A probe、1550 强制 prune；断言 rd_log 事件字段）→ 每个目标：train（train_ckpt_v2 包装，`--saves 12000…29000`）→ val_select_v3（`--delete_unselected`）→ level-1.0（`run_garden.sh FRACS=1.0`）→ 一行结果 `tee -a $TAB`。每阶段可重启（`exists:` 跳过）。
- 提交脚本（`submit-batchN.sh`，heredoc 写出后 `bash`）：行数 + md5 核对所有要用的文件、py_compile、bash -n、数据集存在、squeue 冲突提示、`scontrol show node gl030` 的 AllocTRES，然后 `sbatch --parsable --reservation=yw523-l40s --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,... --job-name=<scene>-<T>-adapt7b $AN/<sbatch>`。
- 场景参数：见 memory/sweep-scene-settings-nov2026.md（XYZ / IMAGES / INDOOR / DATA）。

## 6. 已知坑（都付过学费）

1. `XYZ=auto` 在 sbatch 里坏（`.numpy()` on requires_grad）→ 永远显式 qN（`xyz_rule_v1.py` 算）。
2. upstream `--save_iterations` 无效（只存最终）→ train_ckpt_v2 包装器；中间 checkpoint 没做最终 importance prune → val_select_v3 补做（floor 0.999 / 4×）。
3. `update_indoor(ops)` 在解析后覆盖 CLI 的 add_percentage / max_points → 要改用 train_ckpt_v4 `--override`。
4. indoor 场景必须 `--indoor`（11/01 前所有 run 都漏了）；Blender 全部 `--indoor` + 黑底（交付链写死黑底）。
5. `python - args | tee $TAB <<'PY'` 会把 heredoc 喂给 tee（表为空）→ 必须 `<<'PY' | tee`。
6. open3d 在 HPC 不可用（libEGL）。
7. 节点核心共享 → `--cpus-per-task=6`；squeue 的 `QOSMaxGRESPerUser` 误导。
8. `[ -L x ]` vs `[ -d x ]`：val_select 的 `iteration_30000` 是 symlink。
9. heredoc 粘贴截断：提交脚本按行数核对；`'EOF'` 定界符不能出现在文件里；文件里不能有 ``` 。
10. 同名文件永不覆盖（HPC 上已有的 `train_rd_adapt_v7.py` 不动 → v7b 新文件）。
11. Blender loader 的 Pillow 兼容：`dataset_readers.py:315` 已改 np.uint8。
12. 一个 Blender adapt run ≈ 25 min（训练 ~15 + 16 次 probe ×20 s + val_select + 交付）；Mip-360 adapt run 21–58 min（sweep 实测，20 MB 大场景最长）；固定 λ run 35–51 min 训练 + 8–11 min 协议。v8 每 run 多 7 次 phase-A probe（Blender +2 min，Mip +3–5 min）+ Mip 多 8 轮 phase B（+4–8 min）。
13. 1300-it smoke 的最终 probe 在 5 个 Blender 场景（ficus/hotdog/materials/mic/ship）上挂：1250 轮砍了 30000 对后的退化网格，probe 子进程 rc 1 且 log 为空（harness 问题，真实路径由同场景 4 MB 的 run 证明正常）→ 用 SMOKE=0 重交（node-batch10-v1）。lego/chair/drums 的 smoke 正常。
14. probe 的 K-means（appearance.initialize）要求样本数 ≥ K=4096：kept 顶点少于 4096 的网格 probe 必挂 → v8 phase-A probe 在这种情况下跳过（probe_a_skipped）。

## 7. 现在在跑 / 待提交（11/07 12:00Z）

- `squeue -u zr2245`：应只有 `cpu-torch`（batch11 的 6 个 22158535–43 和 batch12 的 6 个 22161861–66 都已结束）。
- HPC 上已有、不再改：v7b 三件（`train_rd_adapt_v7b.py` 827 行 md5 24f478ca…、`adapt-v7b-blender.sbatch` 139 行、`adapt-v7b.sbatch` 129 行）；v8 三件（`train_rd_adapt_v8.py` 887 行 md5 a150276d…、`adapt-v8-blender.sbatch` 155 行、`adapt-v8.sbatch` 147 行）；**v9 三件（现行）：`$AN/rdgrowth/train_rd_adapt_v9.py`（910 行，md5 a120351f64c9472e0353cef29d8ddb04）、`$AN/adapt-v9-blender.sbatch`（162 行，md5 68a42385114634c478a721a58edf50b6）、`$AN/adapt-v9.sbatch`（154 行，md5 df13ce37898f7dcc34a6c39ccd5d7dba）**；提交脚本 `submit-batch8-v2.sh` … `submit-batch12-v1.sh`。
- 表：`$AN/adapt-v7.txt`（33 行）、`adapt-v7-blender.txt`（16）、`redeliver-v3-blender.txt`（17）、`adapt-v8-blender.txt`（4）、`adapt-v8.txt`（2）、`adapt-v9-blender.txt`（3：lego 2.5、chair 2.5、chair 4）、`adapt-v9.txt`（3：bicycle 10、room 10、room 20）。run 目录 `$S-adapt7-t{5,10,20}mb`、`$S-adapt7b-t{2.5,4}mb(-best)`、`$S-adapt8b-/adapt8-`、`$S-adapt9b-t{T}mb(-best)` / `$S-adapt9-t{T}mb`；.out 在 `$AN/adapt9b-*-adapt9b-*.out` / `$AN/adapt9-*-adapt9-*.out`。
- **已写好、待 Wenhao 选 A 才提交：node-batch13-v1**（`release-adapt-v9/commands/node-batch13-v1.md`）：一次粘贴 = `submit-batch13-v1.sh`（md5 核对三个 v9 文件 + batch12 的 6 个交付存在 + 两张 v9 表 ≥ 3 行，然后 43 个 sbatch：`mip 5; bl25; mip10; bl4; mip20`，batch12 跑过的 6 行不重跑），Mip/T&T `--time=03:30:00`（只有 5 MB 跑 smoke），Blender `--time=01:45:00`（只有 2.5 MB 跑 smoke），≈34 GPU-h。§2 smoke 检查块、§3 早上收结果块（两张表 + 每个 run 的 `[RD 9500]` / 12k probe / Training is done）。
- reservation 到 11/12 09:35。

## 8. 分析侧（本地，非 HPC）

- 项目共享目录 `/mnt/project-files/`（本包是它的快照）；临时文件在 scratchpad；memory 在 `/tmp/claude/memory/team/silo/`（本包 memory/）。
- 分析脚本（CPU，读贴回的表）：`analysis/rd/adapt_analysis_v*.py`、`adapt_replay_v7.py`、`plot_*.py`、`build_11scenes_v*.py`、`report/fig_*.py`（figstyle_v1/v2）、`report/report_tables_v*.py`。
- LaTeX：`analysis/report/latex-v5/main.tex`（本地有 texlive；`build.log`）。

## 9. 接手后让 Wenhao 跑的现场核对块（只读，不会改任何东西）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
scontrol show reservation yw523-l40s | head -3; scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
sacct -X -u $USER -S now-24hours -o JobID,JobName%24,State,Elapsed,ExitCode | tail -30
ls -la $AN/*.sbatch $AN/submit-batch*.sh | tail -20; md5sum $AN/rdgrowth/train_rd_adapt_v7*.py $AN/rdgrowth/train_rd_adapt_v8.py $AN/adapt-v7*.sbatch $AN/adapt-v8*.sbatch 2>/dev/null
ls $AN/*.txt; wc -l $AN/adapt-v7.txt $AN/adapt-v7-blender.txt $AN/adapt-v8*.txt 2>/dev/null; ls -d /scratch/zr2245/*-adapt7-t*mb /scratch/zr2245/*-adapt7b-t*mb* /scratch/zr2245/*-adapt8*-t*mb* 2>/dev/null | head -80
```
