# Grounded Anomaly Reconstruction

Studying the gap between **real and synthetic industrial anomalies** on MVTec AD 2,
in the PatchCore feature space.

**Idea.** For each real defect we (1) caption it with a vision-language model,
(2) inpaint that caption onto a clean image at the real defect's own ground-truth
mask location with a diffusion model, and (3) compare real vs generated vs normal
patches with t-SNE. This tests whether grounding generation in the real defect's
description and location closes the real-vs-synthetic gap.

See `thesis_documents/method.md` for the full method.

## Pipeline

![pipeline](thesis_documents/method_pipeline.png)

## Repository layout

| Path | Role |
|------|------|
| `src/reconstruct_real_defects.py` | Grounded SDXL-inpaint reconstruction loop |
| `src/analysis/tsne_real_vs_gen.py` | Patch-level real-vs-generated t-SNE |
| `src/analysis/plot_pipeline.py` | Pipeline diagram |
| `thesis_documents/method.md` | Method write-up (+ diagram) |
| `llm_captions*.json` | Per-defect captions paired with their source images |

## Usage

    # 1) Reconstruct real defects for a category
    python src/reconstruct_real_defects.py \
        --src_dir data/preprocessed --category can \
        --captions_json llm_captions_can_full.json \
        --out_dirname recon_llmcaption_full --imgs_per_caption 4 --low_vram

    # 2) Compare real vs generated vs normal in t-SNE
    python src/analysis/tsne_real_vs_gen.py \
        --base data/preprocessed --variant recon_llmcaption_full \
        --memseg-variant '' --categories can --out-dir results/recon_tsne_full

The MVTec AD 2 dataset is not included; point `--src_dir` at your preprocessed
copy with `<category>/{train,test}` splits and `*_GT.png` masks.
