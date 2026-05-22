# Fine-tuning the generator (LoRA) — does it close the gap?

The grounded t-SNE study showed the **off-the-shelf** diffusion model can't reproduce
some real defects (`fruit_jelly` fails — generated defects sit among normals). This
experiment tests the next step: **fine-tune the generator on the real defects** and
re-measure.

## Setup
- Category: `fruit_jelly` (the worst off-the-shelf case).
- Base model: **SD 1.5** — the available GPU is 8 GB, which can't fine-tune SDXL.
- A/B: SD 1.5 off-the-shelf (**before**) vs SD 1.5 + LoRA (**after**).
- LoRA: rank 8, 600 steps, trained on 15 real defect crops + their captions.

## Scripts (`src/`)
- `prepare_lora_data.py` — crop each real defect to its mask, write `metadata.jsonl`.
- training: the `diffusers` `train_text_to_image_lora.py` example.
- `generate_lora.py` — generate crops with / without the LoRA (before / after).
- `compare_lora.py` — ResNet50 + t-SNE of real / before / after, and the mean
  distance from generated crops to the nearest real defect.

## Result
Mean distance from generated crops to the nearest real defect:

    before (base SD 1.5) = 14.15   ->   after (SD 1.5 + LoRA) = 12.17   (-14%)

`figures/lora_fruit_jelly_compare.png`: **before (red)** sits in its own region away
from the real defects **(green)**; **after (orange)** relocates onto the real-defect
region.

The before/after image pairs (same prompt + seed, only the LoRA differs) make it
concrete — e.g. `figures/lora_fj_before_0028.png` is a generic red drink in a glass,
while `figures/lora_fj_after_0028.png` is the actual jelly-in-a-plastic-cup with
contamination, matching the dataset.

## Conclusion
Fine-tuning the generator **narrows the gap that grounding alone could not** —
confirming the real-vs-synthetic gap is a generation-model limitation that is
reducible by training the generator on real defects.

## Caveats
- SD 1.5, not SDXL (8 GB GPU limit) — lower base quality.
- Only 15 real examples → the LoRA partly memorises the dataset's overall look
  (overfitting); the shift is real but variety is limited.
- The gap is narrowed, not fully closed.
