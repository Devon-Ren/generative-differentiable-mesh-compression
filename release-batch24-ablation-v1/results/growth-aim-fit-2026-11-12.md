# Can the growth aim be fitted from quantities measured during the scene's own training? (2026-11-12, no new runs)

Data: the 19 no-collapse runs of batch24 (growth plan + byte probe, nothing trimmed afterwards), so the delivered size shows what the aim alone produces. aim_needed = aim_used / hit = the aim that would have landed exactly on the target.

| scene | target | aim used | size error | aim needed | kept share at the last growth probe | final faces per vertex | soup bytes/vertex at the last probe | delivered bytes/vertex |
|---|---|---|---|---|---|---|---|---|
| room | 10 | 1.25 | +8.0% | 1.16 | 0.897 | 2.39 | 7.5 | 11.0 |
| kitchen | 10 | 1.25 | +20.4% | 1.04 | 0.926 | 3.11 | 8.6 | 13.2 |
| counter | 10 | 1.25 | +12.9% | 1.11 | 0.910 | 2.67 | 8.1 | 12.0 |
| bonsai | 10 | 1.25 | +1.5% | 1.23 | 0.919 | 2.34 | 7.8 | 11.0 |
| garden | 10 | 1.25 | +6.4% | 1.17 | 0.961 | 2.49 | 8.7 | 12.0 |
| bicycle | 10 | 1.25 | -1.3% | 1.27 | 0.973 | 2.18 | 8.5 | 11.1 |
| stump | 10 | 1.25 | -2.5% | 1.28 | 0.978 | 2.15 | 8.5 | 10.9 |
| flowers | 10 | 1.25 | -0.4% | 1.26 | 0.950 | 2.12 | 8.3 | 11.0 |
| treehill | 10 | 1.25 | -4.0% | 1.30 | 0.974 | 1.98 | 8.3 | 10.6 |
| truck | 10 | 1.25 | +1.5% | 1.23 | 0.928 | 2.19 | 9.0 | 12.0 |
| train | 10 | 1.25 | +7.3% | 1.16 | 0.920 | 2.38 | 8.4 | 11.8 |
| lego | 2.5 | 1.1 | +12.1% | 0.98 | 0.950 | 3.22 | 14.4 | 18.4 |
| chair | 2.5 | 1.1 | +13.4% | 0.97 | 0.913 | 3.33 | 14.6 | 18.7 |
| drums | 2.5 | 1.1 | +11.2% | 0.99 | 0.931 | 3.31 | 14.1 | 18.1 |
| ficus | 2.5 | 1.1 | -22.2% | 1.41 | 0.961 | 2.79 | 15.9 | 19.2 |
| hotdog | 2.5 | 1.1 | +14.9% | 0.96 | 0.887 | 3.46 | 15.3 | 19.2 |
| materials | 2.5 | 1.1 | +6.9% | 1.03 | 0.952 | 2.79 | 14.0 | 17.3 |
| mic | 2.5 | 1.1 | -6.2% | 1.17 | 0.812 | 2.60 | 15.6 | 18.0 |
| ship | 2.5 | 1.1 | +10.6% | 0.99 | 0.959 | 3.07 | 13.8 | 17.7 |

**Mip-NeRF 360 / T&T (11)**: aim needed 1.04..1.30, mean 1.20, sd 0.077. With the one fixed aim the size error has sd 7.0% (this is before the collapse rounds).
- correlation of aim needed with kept share at the last growth probe: r = +0.67
- correlation of aim needed with soup bytes per vertex at the last probe: r = +0.07
- correlation of aim needed with kept share at the first probe: r = +0.49
- leave-one-scene-out: aim predicted from 'kept' -> size error sd 6.0%, range -6.6..+14.9%, within 10%: 10/11 (assumes the landing scales linearly with the aim; not run)
- leave-one-scene-out: aim predicted from 'sbv' -> size error sd 7.9%, range -8.5..+18.7%, within 10%: 10/11 (assumes the landing scales linearly with the aim; not run)
- reference, one constant (leave-one-out mean of aim needed): size error sd 7.4%, range -8.5..+17.3%, within 10%: 10/11

**NeRF-Synthetic (8)**: aim needed 0.96..1.41, mean 1.06, sd 0.147. With the one fixed aim the size error has sd 12.0% (this is before the collapse rounds).
- correlation of aim needed with kept share at the last growth probe: r = -0.00
- correlation of aim needed with soup bytes per vertex at the last probe: r = +0.71
- correlation of aim needed with kept share at the first probe: r = +0.01
- leave-one-scene-out: aim predicted from 'kept' -> size error sd 20.2%, range -41.2..+13.7%, within 10%: 3/8 (assumes the landing scales linearly with the aim; not run)
- leave-one-scene-out: aim predicted from 'sbv' -> size error sd 13.3%, range -23.3..+24.5%, within 10%: 6/8 (assumes the landing scales linearly with the aim; not run)
- reference, one constant (leave-one-out mean of aim needed): size error sd 13.3%, range -28.3..+12.7%, within 10%: 4/8

## What the aim actually stands for: the faces per vertex of the stitched mesh

- Mip-NeRF 360 / T&T (11): correlation of aim needed with FINAL faces per vertex r = -0.96; if that one number were known during growth, a leave-one-out linear fit gives size error sd 2.7%, range -6.2..+3.5%, within 10%: 11/11 (no collapse; not run)
- NeRF-Synthetic (7, ficus is capacity-limited): correlation of aim needed with FINAL faces per vertex r = -0.89; if that one number were known during growth, a leave-one-out linear fit gives size error sd 5.2%, range -10.8..+7.0%, within 10%: 6/7 (no collapse; not run)
- all 18 together: correlation of aim needed with FINAL faces per vertex r = -0.97; if that one number were known during growth, a leave-one-out linear fit gives size error sd 2.8%, range -3.7..+6.7%, within 10%: 18/18 (no collapse; not run)

The faces per vertex of the final mesh exist only after the restricted Delaunay stitch at 11k, i.e. after growth has ended; the controller plans with a fixed 2.5 until then. So the aim is a stand-in for one unobserved number per scene, and a single relation (aim as a function of faces per vertex) covers captured scenes and synthetic objects alike.

## CPU check (2026-11-12): can a quantity measured DURING growth predict the final faces per vertex, and with it the aim?

82 runs read from rd_log.jsonl (19 no-collapse, 63 full). Growth-time quantities are taken at the last growth probe (iteration 9500).

**Mip-NeRF 360 / T&T (11), no-collapse runs**
- faces per vertex of the cleaned soup: correlation with the final faces per vertex r = -0.70; with the aim needed r = +0.79; leave-one-out aim from it -> size error sd 5.0%, range -8.5..+12.1%, within 10%: 10/11
- faces per vertex of the raw soup: correlation with the final faces per vertex r = -0.74; with the aim needed r = +0.80; leave-one-out aim from it -> size error sd 5.0%, range -8.6..+11.3%, within 10%: 10/11
- kept share: correlation with the final faces per vertex r = -0.51; with the aim needed r = +0.66; leave-one-out aim from it -> size error sd 6.0%, range -6.4..+14.7%, within 10%: 10/11
- reference, one constant: size error sd 7.4%, range -8.4..+17.0%, within 10%: 10/11

**NeRF-Synthetic (7), no-collapse runs**
- faces per vertex of the cleaned soup: correlation with the final faces per vertex r = +0.04; with the aim needed r = -0.36; leave-one-out aim from it -> size error sd 10.0%, range -17.1..+18.9%, within 10%: 5/7
- faces per vertex of the raw soup: correlation with the final faces per vertex r = +0.10; with the aim needed r = -0.29; leave-one-out aim from it -> size error sd 10.2%, range -16.1..+21.6%, within 10%: 5/7
- kept share: correlation with the final faces per vertex r = +0.37; with the aim needed r = -0.73; leave-one-out aim from it -> size error sd 9.6%, range -21.9..+10.5%, within 10%: 5/7
- reference, one constant: size error sd 7.0%, range -15.7..+6.4%, within 10%: 6/7

**all 18, no-collapse runs**
- faces per vertex of the cleaned soup: correlation with the final faces per vertex r = -0.39; with the aim needed r = +0.40; leave-one-out aim from it -> size error sd 11.1%, range -13.9..+19.1%, within 10%: 10/18
- faces per vertex of the raw soup: correlation with the final faces per vertex r = -0.32; with the aim needed r = +0.34; leave-one-out aim from it -> size error sd 11.4%, range -14.0..+18.1%, within 10%: 10/18
- kept share: correlation with the final faces per vertex r = -0.27; with the aim needed r = +0.23; leave-one-out aim from it -> size error sd 12.7%, range -24.7..+17.8%, within 10%: 7/18
- reference, one constant: size error sd 11.3%, range -14.0..+18.5%, within 10%: 8/18

**Is the mesh right after stitching (first probe at 12.5k) already the final one?**
- captured scenes (Mip-360, T&T, Deep Blending), 39 full runs: final faces per vertex / faces per vertex at 12.5k = 0.96..1.56 (mean 1.17); correlation between the two r = +0.70
- NeRF-Synthetic, 24 full runs: final faces per vertex / faces per vertex at 12.5k = 0.99..1.48 (mean 1.24); correlation between the two r = +0.46

indoor example (soup faces per vertex at 9500 -> final): room 0.564 -> 2.39, kitchen 0.565 -> 3.11, counter 0.567 -> 2.67, bonsai 0.565 -> 2.34
