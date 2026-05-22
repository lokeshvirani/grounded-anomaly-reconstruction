# Results — grounded generation vs. real defects

For each category we generate 15 defects (one per distinct real defect), then
embed real-defect / generated / normal image crops with a frozen ResNet50 and
compare them with t-SNE. Figures: `figures/tsne_<category>_crop.png`
(blue = normal, green = real defect, orange = generated).

## Primary evidence: the t-SNE figures

The reliable signal is the figures, read on **two questions**:

1. **Do real defects (green) form clusters separate from normal (blue)?**
   → is the defect even visible in feature space?
2. **Do generated (orange) land on the real-defect clusters?**
   → did generation reproduce it?

Both matter: "generation reproduced the defect" only holds when the answer to (1)
is yes.

## Per-category verdict (read from the figures)

| Category | Real defect visible? | Generated reproduce it? | Verdict |
|---|---|---|---|
| vial | yes (distinct clusters) | yes (orange on several) | **success** |
| can | partly | partly (foil not reproduced) | partial |
| sheet_metal | partly | partly | partial |
| walnuts | yes (distinct clusters) | no (orange among normals) | **failure** |
| fruit_jelly | yes (distinct clusters) | no (orange among normals) | **failure** |
| rice | no (green mixed in blue) | — | inconclusive (defect ≈ normal) |
| fabric | no (green mixed in blue) | — | inconclusive |
| wallplugs | mostly no | — | inconclusive |

## Conclusion

Grounded generation (real caption + real mask) **narrows but does not close** the
gap between real and synthetic defects. Success is **defect-type dependent**: simple
high-contrast additive defects (vial) reproduce; structured / printed-label defects
(can) reproduce only partially; and where the real defect is itself barely
distinguishable from normal (rice, fabric, wallplugs) the comparison is
inconclusive. The limitation is on the **generation** side.

## On quantifying the overlap (why we rely on the figures)

A numeric "overlap score" (fraction of generated defects closer to a real defect
than to a normal) and a companion "separability score" were attempted, but they do
**not give a reliable ranking** — for instructive reasons:

- **Near-duplicate variants.** The dataset has ~6 lighting/shift variants of each
  defect; including them makes nearest-neighbour scores trivially high (a defect's
  nearest neighbour is its own near-identical sibling). Restricting to one image per
  distinct defect fixes this but leaves only 15 points per class.
- **Small, diverse sets.** With 15 diverse defects per class (e.g. vial = particle,
  crack, label, bubbles), the scores are noisy and shift substantially with the
  setup, so they cannot rank categories reliably.

So the numbers are treated as a rough indicator only; the figures and the
two-question reading are the evidence.

## Caveat

Normal crops use a borrowed defect mask (normals have no defect), so the
real-vs-normal separation is approximate; the real-vs-generated comparison in each
figure is the reliable signal. 15 generated images per category.
