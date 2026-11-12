# Vertex cap from a prior vs BudgetMesh as size control (2026-11-12, no new runs)

Bytes per vertex of the encoded file, same codec: BudgetMesh 11.8 (10.1..14.4), MeshSplatting with a vertex cap 11.2 (9.3..14.0). At the same file size the two have the same vertex count within a few percent; the PSNR gain (+0.27 dB same V, +0.21 dB same size) comes from which triangles exist, not from how many.

Size control on the 11 Mip-NeRF 360 / T&T scenes. Cap baselines are read on each scene's measured cap -> file size curve (4-6 real cap runs per scene, log-log interpolation); the prior MB = a*cap + c is fitted on the other 10 scenes (leave-one-scene-out).

| method | trainings | T=5 MB: mean abs err, range, within 10% | T=10 MB | T=20 MB |
|---|---|---|---|---|
| vertex cap from a prior | 1 | 5.6%, -17.9..+5.4%, 9/11 | 5.6%, -5.6..+25.1%, 10/11 | 6.3%, -18.8..+10.5%, 8/11 |
| vertex cap, rescaled after measuring the first file | 2 | 1.6%, -5.9..+0.7%, 11/11 | 1.1%, -4.4..+2.1%, 11/11 | 3.9%, -21.6..+6.6%, 10/11 |
| BudgetMesh | 1 | 3.9%, -4.8..+9.2%, 11/11 | 4.1%, -4.7..+13.5%, 10/11 | 5.1%, -6.6..+10.0%, 10/11 |

Wall-clock per scene on one L40S, training + fine-tune + encode (sacct): BudgetMesh 35 / 42 / 57 min (truck / garden / room); the same without collapse-round probes 26 / 34 / 44 min, which is about what a MeshSplatting cap run costs. BudgetMesh is about 1.3x one cap run; the two-training cap baseline is 2x.

## Does the cap prior transfer to another dataset? (prior fitted on the 11 Mip-NeRF 360 / T&T scenes only: file MB = 7.30 x cap[M] + 2.11)

cap points: {'drjohnson': 3, 'playroom': 3, 'lego': 7, 'chair': 7, 'drums': 7, 'ficus': 7, 'hotdog': 7, 'materials': 7, 'mic': 7, 'ship': 7} 

### Deep Blending (BudgetMesh: settings untouched from Mip-NeRF 360)

- T=5 MB | cap, Mip prior: mean +7.1%, mean abs 7.1%, range +5.4..+8.7%, within 10%: 2/2 | BudgetMesh: mean +7.9%, mean abs 7.9%, range +4.7..+11.1%, within 10%: 1/2
- T=10 MB | cap, Mip prior: mean +27.0%, mean abs 27.0%, range +21.7..+32.4%, within 10%: 0/2 | BudgetMesh: mean +10.2%, mean abs 10.2%, range +8.3..+12.0%, within 10%: 1/2
- T=20 MB | cap, Mip prior: mean +26.9%, mean abs 26.9%, range +12.3..+41.6%, within 10%: 0/2 | BudgetMesh: mean +6.7%, mean abs 6.7%, range +3.6..+9.7%, within 10%: 2/2
- all 6 | cap, Mip prior: mean +20.3%, mean abs 20.3%, range +5.4..+41.6%, within 10%: 2/6 | BudgetMesh: mean +8.2%, mean abs 8.2%, range +3.6..+12.0%, within 10%: 4/6

### NeRF-Synthetic (objects below their capacity limit at that target; BudgetMesh uses the growth aim 1.1 calibrated on this dataset)

- T=1.5 MB, 8 objects | cap, Mip prior: asks for a cap of -84203 vertices (the prior intercept 2.11 MB is above the target): unusable | cap, prior re-fitted on the other 7 objects: mean +1.0%, mean abs 7.0%, range -18.0..+10.6%, within 10%: 6/8 | BudgetMesh: mean +4.0%, mean abs 4.0%, range +1.0..+9.0%, within 10%: 8/8
- T=2.5 MB, 7 objects | cap, Mip prior: mean -43.9%, mean abs 43.9%, range -48.5..-41.3%, within 10%: 0/7 | cap, prior re-fitted on the other 7 objects: mean +6.7%, mean abs 11.7%, range -17.5..+24.2%, within 10%: 3/7 | BudgetMesh: mean +2.6%, mean abs 5.0%, range -6.6..+9.2%, within 10%: 7/7
- T=4 MB, 6 objects | cap, Mip prior: mean -16.5%, mean abs 16.5%, range -30.4..-4.6%, within 10%: 1/6 | cap, prior re-fitted on the other 7 objects: mean -12.5%, mean abs 18.7%, range -32.7..+17.0%, within 10%: 1/6 | BudgetMesh: mean -2.9%, mean abs 4.7%, range -10.3..+3.4%, within 10%: 5/6
