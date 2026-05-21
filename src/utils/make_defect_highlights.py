"""Make defect-highlight thumbnails to caption subtle defects accurately.

For each distinct defect (`bad*_regular.png` + its `_GT` mask), writes a 2-panel
PNG: (left) the full image with the GT-mask bounding box drawn in red, (right) a
zoomed crop of the masked defect region. This lets the captioner see exactly
where and what the real defect is, even in busy textures (fabric, rice, ...).

    python src/utils/make_defect_highlights.py \
        --src_dir data/preprocessed \
        --categories fabric rice sheet_metal vial wallplugs \
        --out_dir results/_defect_highlights
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw

PANEL = 320
PAD_FRAC = 0.30


def make_highlight(img_path, gt_path, out_path):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    m = np.array(Image.open(gt_path).convert("L").resize((W, H), Image.NEAREST)) > 127
    full = img.resize((PANEL, PANEL))
    if m.any():
        ys, xs = np.where(m)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = x1 - x0, y1 - y0
        px, py = int(bw * PAD_FRAC) + 6, int(bh * PAD_FRAC) + 6
        cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
        cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
        crop = img.crop((cx0, cy0, cx1, cy1)).resize((PANEL, PANEL), Image.NEAREST)
        d = ImageDraw.Draw(full)
        sx, sy = PANEL / W, PANEL / H
        d.rectangle([int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)],
                    outline=(255, 0, 0), width=3)
    else:
        crop = full.copy()
    out = Image.new("RGB", (PANEL * 2, PANEL), (255, 255, 255))
    out.paste(full, (0, 0))
    out.paste(crop, (PANEL, 0))
    out.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/preprocessed")
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--out_dir", default="results/_defect_highlights")
    a = ap.parse_args()
    for c in a.categories:
        td = os.path.join(a.src_dir, c, "test")
        od = os.path.join(a.out_dir, c)
        os.makedirs(od, exist_ok=True)
        files = [f for f in sorted(glob.glob(os.path.join(td, "bad*_regular.png")))
                 if "_GT" not in os.path.basename(f)]
        n = 0
        for f in files:
            gt = f.replace(".png", "_GT.png")
            if not os.path.exists(gt):
                continue
            make_highlight(f, gt, os.path.join(od, os.path.basename(f)))
            n += 1
        print(f"{c}: {n} highlights -> {od}")


if __name__ == "__main__":
    main()
