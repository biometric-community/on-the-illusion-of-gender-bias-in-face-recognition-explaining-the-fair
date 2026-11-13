# DEVIATIONS — Illusion of Gender Bias

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Attribute corpus | MAAD-Face lineage | CelebA 40 attrs (Male + 39 ND) | Local public labels |
| D2 | FRS | Paper commercial/deep FRS | ResNet-18 CE lab twin | Stack consistency |
| D3 | Fairness metrics | iGARBE on FMR/FNMR at FMR = 1e-3 | iGARBE at FMR = 1e-3 (alpha = 0.5, not stated in the extract) ranks the filters; accuracy gap, AFI and each gender's FNMR are reported alongside | Same metric as the paper |
| D4 | Eval identities | MAAD / VGGFace2 scale | CelebA val probe pairs + LFW family | Data availability |
