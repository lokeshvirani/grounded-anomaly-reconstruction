# Results — grounded generation vs. real defects (t-SNE)

For each category we generate 15 defects (one per distinct real defect), then
embed real-defect / generated / normal image crops with a frozen ResNet50 and
compare them with t-SNE. Figures are in `figures/tsne_<category>_crop.png`
(each: blue = normal, green = real defect, orange = generated).

## Overlap score

Fraction of generated defects whose nearest real defect is closer than their
nearest normal, with normals subsampled to match the number of real defects
(averaged over 20 subsamples), computed on the raw 2048-d features.
~1.0 = generated look like real defects; ~0.0 = they look like normal images.

| Category | Overlap score |
|---|---|
| rice | 0.54 |
| wallplugs | 0.43 |
| fabric | 0.40 |
| vial | 0.32 |
| sheet_metal | 0.29 |
| can | 0.16 |
| fruit_jelly | 0.12 |
| walnuts | 0.10 |

## How to read the figures (two questions)

The overlap score is only meaningful when the real defects are themselves
separable from normal, so read each figure on two questions:

1. Do real defects (green) form clusters separate from normal (blue)?
   → is the defect even visible in feature space?
2. Do generated (orange) land on the real-defect clusters?
   → did generation reproduce it?

- **Genuine partial success:** `vial` — real defects are distinct and generated
  land on several of the real clusters.
- **Genuine failures:** `walnuts`, `fruit_jelly`, `can` — real defects are
  distinct, but generated land among the normals.
- **Inconclusive (defect ~ normal):** `rice`, `fabric`, `wallplugs` — real
  defects barely separate from normal, so a high score reflects intermixing,
  not successful reproduction.

## Conclusion

Grounded generation (real caption + real mask) narrows but does not close the
gap between real and synthetic defects. Success is defect-type dependent, and
even in the best case (`vial`) the generated defects only partially match the
real ones — the limitation is on the generation side.

Caveat: normal crops use a borrowed defect mask (normals have no defect), so the
real-vs-normal separation is approximate; the real-vs-generated comparison is the
reliable signal. Only 15 generated images per category.
