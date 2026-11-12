# batch27: three intuitive size-control baselines, one scene per dataset (measured, 2026-11-12)

Targets: garden, truck, playroom 10 MB; lego 2.5 MB. Size error = file size / target - 1. dPSNR = BudgetMesh minus the baseline at the baseline's file size (BudgetMesh read on its own measured targets, log-linear).

| scene | BudgetMesh (1 training) | cap from a prior (1 training) | cap rescaled and retrained (2 trainings) | probe-driven count + MeshSplatting selection (1 training) |
|---|---|---|---|---|
| garden | 10.21 MB (+2.0%), 23.61 dB | 10.30 MB (+3.0%), 23.49 dB, dPSNR +0.13 | 11.64 MB (+16.4%), 23.65 dB, dPSNR +0.09 | 9.83 MB (-1.7%), 23.40 dB, dPSNR +0.17 |
| truck | 10.12 MB (+1.2%), 22.17 dB | 9.95 MB (-0.5%), 22.16 dB, dPSNR -0.00 | 10.00 MB (+0.0%), 22.19 dB, dPSNR -0.03 | 9.28 MB (-7.2%), 21.98 dB, dPSNR +0.14 |
| playroom | 10.83 MB (+8.3%), 26.98 dB | 10.96 MB (+9.6%), 27.08 dB, dPSNR -0.09 | 11.63 MB (+16.3%), 27.00 dB, dPSNR +0.04 | 11.20 MB (+12.0%), 26.94 dB, dPSNR +0.07 |
| lego | 2.50 MB (-0.2%), 32.38 dB | 1.45 MB (-41.8%), 29.77 dB, dPSNR +0.73 | 1.97 MB (-21.4%), 31.04 dB, dPSNR +0.51 | 2.74 MB (+9.6%), 32.01 dB, dPSNR +0.61 |
| within 10% | 4/4, mean abs 2.9% | 3/4, mean abs 13.7%, mean dPSNR +0.19 | 1/4, mean abs 13.5%, mean dPSNR +0.15 | 3/4, mean abs 7.6%, mean dPSNR +0.25 |

## Why the second cap training did not help: the cap is a loose handle on the final vertex count

- garden: cap 1,105,000 -> 884,328 vertices (V/cap 0.80); cap 1,073,000 -> 1,005,902 vertices (V/cap 0.94)
- truck: cap 1,078,000 -> 885,368 vertices (V/cap 0.82); cap 1,083,000 -> 893,100 vertices (V/cap 0.82)
- playroom: cap 1,081,000 -> 847,103 vertices (V/cap 0.78); cap 986,000 -> 937,937 vertices (V/cap 0.95)
- lego: cap 53,000 -> 55,165 vertices (V/cap 1.04); cap 91,000 -> 92,381 vertices (V/cap 1.02)

garden and playroom: the second training used a SMALLER cap and ended with MORE vertices and a larger file.
