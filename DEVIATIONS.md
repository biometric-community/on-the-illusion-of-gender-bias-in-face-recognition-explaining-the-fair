# DEVIATIONS — Illusion of Gender Bias

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Attribute corpus | MAAD-Face lineage | CelebA 40 attrs (Male + 39 ND) | Local public labels |
| D2 | FRS | Paper commercial/deep FRS | ResNet-18 CE lab twin | Stack consistency |
| D3 | Fairness metrics | iGARBE on FMR/FNMR at FMR = 1e-3 | iGARBE at FMR = 1e-3 (alpha = 0.5, not stated in the extract) alongside the accuracy gap and absolute fairness impact (AFI) used to rank filters | Filters stay ranked by the accuracy gap |
| D4 | Eval identities | MAAD / VGGFace2 scale | CelebA val probe pairs + LFW family | Data availability |
