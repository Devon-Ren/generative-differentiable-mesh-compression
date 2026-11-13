# batch30 analysis (collect of 2026-11-12 20:07 EST)

```
## current method at the same targets
  Mip-360 / T&T  n 11 | mean abs  4.1% | within 2%:  4  5%:  8  10%: 10 | range  -4.7% .. +13.5%
  Deep Blending  n  2 | mean abs 10.1% | within 2%:  0  5%:  0  10%:  1 | range  +8.3% .. +11.9%
  objects        n  8 | mean abs  7.1% | within 2%:  2  5%:  3  10%:  7 | range -21.6% ..  +9.2%
  all            n 21 | mean abs  5.8% | within 2%:  6  5%: 11  10%: 18 | range -21.6% .. +13.5%

## variant final (21 of 21 scenes)
  scene            MB     err    PSNR | current:     err    PSNR | PSNR new - current at the new file size | SSIM / LPIPS new vs current run
  room         10.010   +0.1%  27.954 |   +2.9%  27.833 | +0.13 dB | 0.8457 / 0.4230 vs 0.8371 / 0.4353
  kitchen      10.069   +0.7%  26.444 |  +13.5%  26.423 | +0.18 dB | 0.8184 / 0.3495 vs 0.8165 / 0.3530
  counter      10.059   +0.6%  25.763 |   +7.9%  25.659 | +0.15 dB | 0.8054 / 0.4271 vs 0.7949 / 0.4402
  bonsai        9.988   -0.1%  26.802 |   +0.9%  26.972 | -0.14 dB | 0.8303 / 0.4584 vs 0.8275 / 0.4638
  garden       10.045   +0.4%  23.688 |   +2.0%  23.610 | +0.10 dB | 0.6715 / 0.3807 vs 0.6645 / 0.3887
  bicycle       9.999   -0.0%  22.487 |   -1.6%  22.466 | +0.01 dB | 0.5728 / 0.4864 vs 0.5673 / 0.4919
  stump         9.936   -0.6%  23.960 |   -3.6%  23.943 | -0.00 dB | 0.6163 / 0.4592 vs 0.6121 / 0.4655
  flowers      10.013   +0.1%  19.113 |   -1.3%  19.018 | +0.09 dB | 0.4490 / 0.5058 vs 0.4437 / 0.5106
  treehill      9.959   -0.4%  20.558 |   -4.7%  20.463 | +0.10 dB | 0.5034 / 0.5721 vs 0.4976 / 0.5768
  truck         9.990   -0.1%  22.274 |   +1.2%  22.174 | +0.11 dB | 0.7820 / 0.3119 vs 0.7784 / 0.3187
  train        10.073   +0.7%  18.781 |   +5.1%  18.730 | +0.06 dB | 0.6868 / 0.4127 vs 0.6835 / 0.4169
  drjohnson    10.047   +0.5%  26.151 |  +11.9%  26.091 | +0.16 dB | 0.8192 / 0.5048 vs 0.8170 / 0.5057
  playroom      9.635   -3.6%  27.235 |   +8.3%  26.982 | +0.36 dB | 0.8484 / 0.4432 vs 0.8432 / 0.4498
  lego          2.490   -0.4%  31.890 |   -0.2%  32.383 | -0.48 dB | 0.9566 / 0.0620 vs 0.9623 / 0.0527  (checkpoint 28000)
  chair         2.498   -0.1%  31.195 |   -1.7%  32.259 | -1.09 dB | 0.9595 / 0.0565 vs 0.9685 / 0.0445  (checkpoint 28000)
  drums         2.500   +0.0%  25.403 |   +7.0%  25.484 | -0.05 dB | 0.9382 / 0.0749 vs 0.9403 / 0.0724  (checkpoint 28000)
  ficus         1.934  -22.6%  31.849 |  -21.6%  32.110 | -0.25 dB | 0.9689 / 0.0444 vs 0.9713 / 0.0416  (checkpoint 28000)
  hotdog        2.500   +0.0%  34.856 |   +9.2%  35.432 | -0.53 dB | 0.9703 / 0.0706 vs 0.9739 / 0.0612  (checkpoint 28000)
  materials     2.504   +0.2%  27.605 |   +4.0%  27.793 | -0.17 dB | 0.9234 / 0.1139 vs 0.9297 / 0.1043  (checkpoint 28000)
  mic           2.490   -0.4%  32.417 |   -6.6%  33.303 | -0.99 dB | 0.9784 / 0.0367 vs 0.9832 / 0.0271  (checkpoint 28000)
  ship          2.494   -0.2%  29.012 |   +6.4%  29.013 | +0.08 dB | 0.8705 / 0.1703 vs 0.8702 / 0.1717  (checkpoint 28000)
  Mip-360 / T&T  n 11 | mean abs  0.4% | within 2%: 11  5%: 11  10%: 11 | range  -0.6% ..  +0.7% | PSNR at the same size: mean +0.07 dB, better on 9 of 11
  Deep Blending  n  2 | mean abs  2.1% | within 2%:  1  5%:  2  10%:  2 | range  -3.6% ..  +0.5% | PSNR at the same size: mean +0.26 dB, better on 2 of 2
  objects        n  8 | mean abs  3.0% | within 2%:  7  5%:  7  10%:  7 | range -22.6% ..  +0.2% | PSNR at the same size: mean -0.44 dB, better on 1 of 8
  all            n 21 | mean abs  1.5% | within 2%: 19  5%: 20  10%: 20 | range -22.6% ..  +0.7% | PSNR at the same size: mean -0.10 dB, better on 12 of 21

## variant ratio35 (19 of 21 scenes)
  scene            MB     err    PSNR | current:     err    PSNR | PSNR new - current at the new file size | SSIM / LPIPS new vs current run
  room          9.690   -3.1%  27.888 |   +2.9%  27.833 | +0.08 dB | 0.8425 / 0.4280 vs 0.8371 / 0.4353
  kitchen       9.700   -3.0%  26.361 |  +13.5%  26.423 | +0.15 dB | 0.8156 / 0.3540 vs 0.8165 / 0.3530
  counter       9.562   -4.4%  25.656 |   +7.9%  25.659 | +0.07 dB | 0.8011 / 0.4335 vs 0.7949 / 0.4402
  bonsai        9.335   -6.6%  27.027 |   +0.9%  26.972 | +0.26 dB | 0.8303 / 0.4605 vs 0.8275 / 0.4638
  garden        9.649   -3.5%  23.655 |   +2.0%  23.610 | +0.11 dB | 0.6679 / 0.3852 vs 0.6645 / 0.3887
  bicycle       9.724   -2.8%  22.547 |   -1.6%  22.466 | +0.09 dB | 0.5733 / 0.4866 vs 0.5673 / 0.4919
  stump         9.433   -5.7%  23.986 |   -3.6%  23.943 | +0.06 dB | 0.6157 / 0.4597 vs 0.6121 / 0.4655
  flowers       9.558   -4.4%  19.091 |   -1.3%  19.018 | +0.08 dB | 0.4452 / 0.5082 vs 0.4437 / 0.5106
  treehill      9.676   -3.2%  20.472 |   -4.7%  20.463 | +0.01 dB | 0.5003 / 0.5732 vs 0.4976 / 0.5768
  truck         9.931   -0.7%  22.225 |   +1.2%  22.174 | +0.06 dB | 0.7811 / 0.3139 vs 0.7784 / 0.3187
  train         9.766   -2.3%  18.716 |   +5.1%  18.730 | +0.00 dB | 0.6851 / 0.4148 vs 0.6835 / 0.4169
  drjohnson     9.198   -8.0%  26.039 |  +11.9%  26.091 | +0.12 dB | 0.8191 / 0.5063 vs 0.8170 / 0.5057
  playroom      9.093   -9.1%  27.141 |   +8.3%  26.982 | +0.32 dB | 0.8478 / 0.4458 vs 0.8432 / 0.4498
  lego          2.398   -4.1%  32.645 |   -0.2%  32.383 | +0.40 dB | 0.9639 / 0.0504 vs 0.9623 / 0.0527  (checkpoint 22000)
  chair         2.306   -7.8%  32.122 |   -1.7%  32.259 | -0.00 dB | 0.9678 / 0.0448 vs 0.9685 / 0.0445  (checkpoint 21000)
  drums         2.431   -2.8%  25.514 |   +7.0%  25.484 | +0.07 dB | 0.9407 / 0.0718 vs 0.9403 / 0.0724  (checkpoint 26000)
  ficus         1.964  -21.4%  32.117 |  -21.6%  32.110 | -0.01 dB | 0.9712 / 0.0416 vs 0.9713 / 0.0416  (checkpoint 26000)
  hotdog        2.449   -2.0%  35.373 |   +9.2%  35.432 | +0.00 dB | 0.9734 / 0.0621 vs 0.9739 / 0.0612  (checkpoint 26000)
  mic           2.346   -6.2%  33.372 |   -6.6%  33.303 | +0.06 dB | 0.9835 / 0.0266 vs 0.9832 / 0.0271  (checkpoint 21000)
  Mip-360 / T&T  n 11 | mean abs  3.6% | within 2%:  1  5%:  9  10%: 11 | range  -6.6% ..  -0.7% | PSNR at the same size: mean +0.09 dB, better on 11 of 11
  Deep Blending  n  2 | mean abs  8.5% | within 2%:  0  5%:  0  10%:  2 | range  -9.1% ..  -8.0% | PSNR at the same size: mean +0.22 dB, better on 2 of 2
  objects        n  6 | mean abs  7.4% | within 2%:  0  5%:  3  10%:  5 | range -21.4% ..  -2.0% | PSNR at the same size: mean +0.09 dB, better on 4 of 6
  all            n 19 | mean abs  5.3% | within 2%:  1  5%: 12  10%: 18 | range -21.4% ..  -0.7% | PSNR at the same size: mean +0.10 dB, better on 17 of 19

## variant ratio (1 of 21 scenes)
  scene            MB     err    PSNR | current:     err    PSNR | PSNR new - current at the new file size | SSIM / LPIPS new vs current run
  kitchen       9.768   -2.3%  26.363 |  +13.5%  26.423 | +0.14 dB | 0.8134 / 0.3567 vs 0.8165 / 0.3530
  Mip-360 / T&T  n  1 | mean abs  2.3% | within 2%:  0  5%:  1  10%:  1 | range  -2.3% ..  -2.3% | PSNR at the same size: mean +0.14 dB, better on 1 of 1
  all            n  1 | mean abs  2.3% | within 2%:  0  5%:  1  10%:  1 | range  -2.3% ..  -2.3% | PSNR at the same size: mean +0.14 dB, better on 1 of 1

## cap baseline (cap from the leave-one-out prior; one retry on a miss > 10%)
  scene            cap     err    PSNR | retry cap               err | trainings to +-10% | our final design: err
  room         1078000   +6.7%  27.727 |                             | 1 | +0.1%
  kitchen      1086000  +16.3%  25.928 |              900000   -9.8% | 2 | +0.7%
  counter      1083000   -6.9%  25.318 |                             | 1 | +0.6%
  bonsai       1073000  -16.9%  26.665 |             1367000   -1.4% | 2 | -0.1%
  garden       1105000   +3.0%  23.490 |                             | 1 | +0.4%
  bicycle      1080000   +0.4%  22.095 |                             | 1 | -0.0%
  stump        1069000   +2.1%  23.629 |                             | 1 | -0.6%
  flowers      1071000   -5.8%  18.622 |                             | 1 | +0.1%
  treehill     1079000   +6.0%  20.487 |                             | 1 | -0.4%
  truck        1078000   -0.5%  22.165 |                             | 1 | -0.1%
  train        1085000   +3.6%  18.514 |                             | 1 | +0.7%
  drjohnson    1081000  +15.7%  26.165 |              902000  +16.9% | >2 | +0.5%
  playroom     1081000   +9.6%  27.083 |                             | 1 | -3.6%
  lego          183000  +11.6%  32.314 |              154000  +11.7% | >2 | -0.4%
  chair         178000   +5.4%  32.070 |                             | 1 | -0.1%
  drums         176000   +9.5%  25.417 |                             | 1 | +0.0%
  ficus         171000  -39.6%  31.899 |              515000  -40.0% | >2 | -22.6%
  hotdog        177000   +9.9%  35.571 |                             | 1 | +0.0%
  materials     176000   +1.4%  27.759 |                             | 1 | +0.2%
  mic           174000  -18.4%  33.001 |              247000  -18.0% | >2 | -0.4%
  ship          177000   +2.9%  28.766 |                             | 1 | -0.2%
  lego third training (cap 130000): +0.2%
  first training: n 21 | mean abs  9.1% | within 2%:  3  5%:  7  10%: 15 | range -39.6% .. +16.3%
  after the retry: n 21 | mean abs  8.2% | within 2%:  4  5%:  8  10%: 17 | range -40.0% .. +16.9% | scenes within 10% after 1 / 2 trainings: 15 / 17 of 21

## job duration in minutes (whole job: training, fine-tuning, encoding; cap jobs also run a control stage)
  final    Mip-360 / T&T  n 11  mean  47.2  range  36.0 ..  60.8
  final    Deep Blending  n  2  mean  42.9  range  41.5 ..  44.4
  final    objects        n  8  mean  25.7  range  23.4 ..  26.7
  ratio35  Mip-360 / T&T  n 11  mean  44.8  range  33.5 ..  59.3
  ratio35  Deep Blending  n  2  mean  41.2  range  39.8 ..  42.5
  ratio35  objects        n  6  mean  26.3  range  25.0 ..  27.9
  cap1     Mip-360 / T&T  n  8  mean  42.8  range  33.4 ..  53.6
  cap1     Deep Blending  n  1  mean  48.1  range  48.1 ..  48.1
  cap1     objects        n  6  mean  23.5  range  20.2 ..  26.2

final design minus vertex cap at the same file size (cap curve interpolated between the two nearest cap runs of the scene)
  room        10.010 MB  PSNR +0.29 dB  SSIM +0.0095  LPIPS -0.0170
  kitchen     10.069 MB  PSNR +0.70 dB  SSIM +0.0308  LPIPS -0.0413
  counter     10.059 MB  PSNR +0.42 dB  SSIM +0.0150  LPIPS -0.0190
  bonsai       9.988 MB  PSNR -0.03 dB  SSIM +0.0055  LPIPS -0.0095
  garden      10.045 MB  PSNR +0.21 dB  SSIM +0.0078  LPIPS -0.0162
  bicycle      9.999 MB  PSNR +0.38 dB  SSIM +0.0109  LPIPS -0.0089
  stump        9.936 MB  PSNR +0.35 dB  SSIM +0.0232  LPIPS -0.0268
  flowers     10.013 MB  PSNR +0.41 dB  SSIM +0.0258  LPIPS -0.0321
  treehill     9.959 MB  PSNR +0.11 dB  SSIM +0.0081  LPIPS -0.0083
  truck        9.990 MB  PSNR +0.09 dB  SSIM +0.0017  LPIPS -0.0081
  train       10.073 MB  PSNR +0.26 dB  SSIM +0.0090  LPIPS -0.0145
  drjohnson   10.047 MB  PSNR +0.11 dB  SSIM +0.0073  LPIPS -0.0056
  playroom     9.635 MB  PSNR +0.35 dB  SSIM +0.0079  LPIPS -0.0109
  lego         2.490 MB  PSNR -0.12 dB  SSIM -0.0024  LPIPS +0.0037
  chair        2.498 MB  PSNR -0.77 dB  SSIM -0.0066  LPIPS +0.0090
  drums        2.500 MB  PSNR -0.01 dB  SSIM -0.0007  LPIPS +0.0007
  ficus      1.934 MB: outside the range of the cap runs (1.02, 1.33, 1.35, 1.49, 1.49, 1.50, 1.50, 1.51, 1.51)
  hotdog       2.500 MB  PSNR -0.59 dB  SSIM -0.0048  LPIPS +0.0129
  materials    2.504 MB  PSNR -0.15 dB  SSIM -0.0060  LPIPS +0.0073
  mic        2.490 MB: outside the range of the cap runs (1.26, 1.48, 1.61, 2.04, 2.04, 2.05, 2.05, 2.06, 2.07)
  ship         2.494 MB  PSNR +0.31 dB  SSIM +0.0047  LPIPS -0.0082
  Mip-360 / T&T  n 11  PSNR mean +0.29 dB (better on 10)  SSIM +0.0134  LPIPS -0.0183
  Deep Blending  n  2  PSNR mean +0.23 dB (better on 2)  SSIM +0.0076  LPIPS -0.0082
  objects        n  6  PSNR mean -0.22 dB (better on 1)  SSIM -0.0026  LPIPS +0.0042

time of one training in minutes, same scenes (cap: training + fine-tuning + encoding from the stage time stamps; ours: the whole job)
  Mip-360 / T&T  n 9  cap 34.0  final design 49.0  (ratio 1.44)  scenes: room kitchen counter bonsai bicycle stump flowers treehill train
  Deep Blending  n 1  cap 31.3  final design 44.4  (ratio 1.42)  scenes: drjohnson
  cap stage A minutes: {'bicycle': 28.6, 'bonsai': 41.1, 'counter': 42.4, 'drjohnson': 31.3, 'flowers': 28.7, 'kitchen': 42.7, 'room': 41.2, 'stump': 27.2, 'train': 24.8, 'treehill': 29.1}
```
