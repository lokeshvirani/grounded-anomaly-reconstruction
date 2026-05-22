# PIPELINE — quick reference 

**One sentence:** For each real defect I take its vision-LLM caption and its real
GT mask, inpaint that defect onto a clean image with a diffusion model, then embed
the real / generated / normal defect-regions with ResNet50 and compare them with
t-SNE plus an overlap score.

## The 4 files

| File | One-line role |
|---|---|
| `src/caption.py` | real defect image → one-sentence defect caption (vision LLM). *(kept local; captions are cached in the JSONs)* |
| `src/regenerate.py` | good image + mask + caption → generated defect (SDXL inpainting) |
| `src/run_pipeline.py` | orchestrator: for each defect, load caption + mask + a good base, call regenerate, save |
| `src/embed_tsne.py` | embed real / generated / normal (ResNet50) → t-SNE plot + overlap score |

## Two models — both pretrained (not trained by me)

| Role | Model | Trained by me? |
|---|---|---|
| **Generate** the defect | SDXL inpainting (`stable-diffusion-xl-1.0-inpainting-0.1`) | No — used as-is |
| **Compare** real vs generated | ResNet50 (ImageNet, classifier head removed) | No — used as-is |

The SDXL inpainting model *creates* the defect; ResNet50 generates nothing — it only
turns images into feature vectors for the t-SNE comparison. Because the generator is
used pretrained/off-the-shelf, the "generation limitation" finding is the limit of the
**off-the-shelf** model — the natural next step to close the gap would be to *fine-tune*
the generator on real defects.

## How it runs, step by step

**Generate (`run_pipeline.py` + `regenerate.py`):**
1. Load captions from `llm_captions_<cat>_full.json` → `(caption, source)` pairs.
2. Load the SDXL inpainting model onto the GPU (once, before the loop).
3. For each defect: find the real bad image + its GT mask (the defect *location*).
4. Pick a random good (normal) base image.
5. Prep inputs: good image → RGB 256×256; mask → grayscale, soft edges.
6. Inpaint: `prompt` = caption (**what**), `mask` = **where**, `image` = good base;
   plus a negative prompt and a fixed seed (repeatable).
7. Save the generated image, its mask, and a `good | real | generated` strip.

**Evaluate (`embed_tsne.py`):**
8. Collect 3 groups: real defects, normals, generated.
9. (`--crop_to_mask`) crop each image to its defect region.
10. Frozen ResNet50 → a 2048-number vector per image.
11. Overlap score: is each generated nearer a real defect or a normal? (counts balanced).
12. t-SNE → 2-D plot: blue = normal, green = real, orange = generated.

## Key design choices (and why)

- **Captions are cached** in JSON (made by a vision LLM earlier) → reproducible, no
  API key. The caption *is* passed to the diffusion model as the prompt.
- **Bad image = reference, not input.** Generation is grounded by the bad image's
  caption + its mask location; resemblance is then *measured* by t-SNE (not img2img).
- **Defect crop before embedding** → the comparison is about the defect, not the
  whole product (whole-image t-SNE just showed a product-pose "ring").
- **Per-category negative prompt only for `can`** (its defect IS printed text/foil,
  so the default "no text" was suppressing the real defect).
- **Balanced overlap score** → normals outnumber defects, so we subsample normals to
  match the count and average over 20 draws.

## Results (full table in `RESULTS.md`)

Overlap score, high → low:
`rice 0.54 · wallplugs 0.43 · fabric 0.40 · vial 0.32 · sheet_metal 0.29 · can 0.16 · fruit_jelly 0.12 · walnuts 0.10`

**Read each figure on two questions:**
1. Do green (real) points separate from blue (normal)? → *is the defect visible?*
2. Do orange (generated) land on the green? → *did generation reproduce it?*

- **vial** = genuine partial success · **walnuts / fruit_jelly / can** = genuine
  failure · **rice / fabric / wallplugs** = inconclusive (defect ≈ normal, so the
  high score is intermixing, not success).

**Headline:** Grounded generation narrows but does **not** close the real-vs-synthetic
gap; success is defect-type dependent → it's a **generation-model** limitation.

## Likely questions

- *"Does it use the LLM caption?"* → Yes — the caption is the diffusion prompt (cached in JSON).
- *"Why crop to the mask?"* → whole-image embeddings are dominated by product pose, not the small defect.
- *"What does 'use the bad image as target' mean here?"* → grounding (caption + mask) + measuring resemblance via t-SNE, not an img2img/loss.
- *"Why is rice's score high?"* → its real defects barely differ from normal, so the score is inflated — that's why I read it with the two-question method, not the number alone.
