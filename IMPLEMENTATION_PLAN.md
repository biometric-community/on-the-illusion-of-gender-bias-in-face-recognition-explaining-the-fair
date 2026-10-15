# IMPLEMENTATION_PLAN — Illusion of Gender Bias

## Paper

Gender gap in FR largely tracks non-demographic appearance attributes; unsupervised filters can make the gap vanish.

## Lab twin

| Component | Choice |
|-----------|--------|
| Backbone | ResNet-18 CE on CelebA |
| Attributes | CelebA 39 non-demographic + Male |
| Decorrelation | Pearson corr + greedy clustering |
| Gap | Acc(MM) − Acc(FF) on probe pairs |
| Search | Share attr=0/1 (and pairwise) filters ranked by AFI |
