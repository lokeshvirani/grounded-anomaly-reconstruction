"""Compare real defect crops vs SD1.5 crops before-LoRA vs after-LoRA.

Embeds each set with ResNet50, plots them in one t-SNE, and reports the mean
distance from generated crops to the nearest real crop (before vs after).
If 'after' is closer than 'before', fine-tuning moved the generated defects
toward the real ones = it narrowed the gap.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from embed_tsne import load_feature_extractor, embed_images   # reuse our own code


def list_pngs(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def mean_nn_dist(gen_feats, real_feats):
    """Average distance from each generated crop to its nearest real crop."""
    return float(np.mean([np.linalg.norm(real_feats - g, axis=1).min() for g in gen_feats]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)     # lora_data/<cat> = real defect crops
    ap.add_argument("--before_dir", required=True)   # base SD1.5 generations
    ap.add_argument("--after_dir", required=True)    # SD1.5 + LoRA generations
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    real = list_pngs(args.real_dir)
    before = list_pngs(args.before_dir)
    after = list_pngs(args.after_dir)
    print(f"real={len(real)} before={len(before)} after={len(after)}")

    model = load_feature_extractor()
    real_f = embed_images(model, real)
    before_f = embed_images(model, before)
    after_f = embed_images(model, after)

    print(f"mean dist to nearest real:  before={mean_nn_dist(before_f, real_f):.2f}  "
          f"after={mean_nn_dist(after_f, real_f):.2f}   (lower = closer to real)")

    feats = np.concatenate([real_f, before_f, after_f])
    perp = min(30, len(feats) - 1)
    xy = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(feats)
    nr, nb = len(real_f), len(before_f)
    r_xy, b_xy, a_xy = xy[:nr], xy[nr:nr + nb], xy[nr + nb:]

    plt.figure(figsize=(8, 8))
    plt.scatter(b_xy[:, 0], b_xy[:, 1], c="red", label="before (base SD1.5)", alpha=0.6)
    plt.scatter(a_xy[:, 0], a_xy[:, 1], c="orange", label="after (SD1.5 + LoRA)", alpha=0.6)
    plt.scatter(r_xy[:, 0], r_xy[:, 1], c="green", label="real defect", alpha=0.9)
    plt.legend()
    plt.title("LoRA fine-tune: before vs after vs real defects")
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
