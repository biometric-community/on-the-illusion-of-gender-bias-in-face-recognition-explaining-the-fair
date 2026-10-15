# REPORT — Illusion of Gender Bias via Non-demographic Attributes

**Protocol:** `full`

Lab twin of Kurz et al.: decorrelated CelebA attributes, gender-gap on
same/different probe pairs, and unsupervised vanishing-bias filter search.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Non-demographic attrs: 39
- Elapsed (s): 1161.8

![Training](outputs/figures/full/train_loss.svg)

## Verification utility

| Dataset | Accuracy | EER | N |
|---------|----------|-----|---|
| calfw | 51.55% | 48.45% | 6000 |
| cplfw | 53.88% | 46.12% | 6000 |
| lfw | 60.65% | 39.35% | 6000 |

## Gender gap (baseline)

- Acc FF: **0.6899** (n=1777)
- Acc MM: **0.7640** (n=1250)
- Gap (MM−FF): **0.0741**
- Attr clusters (|corr|≥0.5): 33

![Gap](outputs/figures/full/gender_gap.svg)

## Vanishing-bias filters

- Best filter: `Mouth_Slightly_Open=1 & Pointy_Nose=1`
- Filtered gap: **0.0000** (AFI=0.0741, n=112)

| Filter | |Gap| | AFI | N |
|--------|------|-----|---|
| Mouth_Slightly_Open=1 & Pointy_Nose=1 | 0.0000 | 0.0741 | 112 |
| Big_Lips=1 & Pointy_Nose=1 | 0.0030 | 0.0710 | 61 |
| Big_Lips=1 & Wavy_Hair=1 | 0.0034 | 0.0707 | 130 |
| Big_Lips=1 & Oval_Face=1 | 0.0080 | 0.0661 | 43 |
| Big_Nose=1 & No_Beard=1 | 0.0085 | 0.0655 | 222 |
| Pointy_Nose=1 & Smiling=1 | 0.0104 | 0.0637 | 113 |
| Attractive=1 & Big_Nose=1 | 0.0111 | 0.0630 | 44 |
| Attractive=1 & Big_Lips=1 | 0.0176 | 0.0565 | 195 |

![Filters](outputs/figures/full/vanishing_filters.svg)

## Notes

- Paper uses MAAD-Face lineage; this rebuild uses CelebA 39 non-demographic attrs + Male.
- See `DEVIATIONS.md`.
