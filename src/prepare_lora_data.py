"""Prepare a LoRA training set: crop each real defect to its mask, pair with its caption.

Output (the format the diffusers LoRA trainer reads):
    <out_dir>/0000.png, 0001.png, ...          # defect crops, resized square
    <out_dir>/metadata.jsonl                    # one {"file_name","text"} per line
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def find_mask(image_path):
    """foo.png -> foo_GT.png (the mask sits next to the image)."""
    stem, ext = os.path.splitext(image_path)
    return stem + "_GT" + ext


def resolve_source(cat_dir, source_rel):
    """The captions JSON uses test/ paths, but files may live under train/. Try both."""
    candidates = [source_rel]
    for a, b in (("test/", "train/"), ("train/", "test/")):
        if source_rel.startswith(a):
            candidates.append(b + source_rel[len(a):])
    for rel in candidates:
        full = os.path.join(cat_dir, rel)
        if os.path.exists(full):
            return full
    return None


def crop_to_mask(img, mask_path, pad=8):
    """Crop img to the bounding box of the defect (white area of the mask)."""
    mask = np.array(Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST))
    ys, xs = np.where(mask > 10)
    if len(xs) == 0:
        return img
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
    return img.crop((x0, y0, x1, y1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="dataset/preprocessed")
    ap.add_argument("--category", required=True)
    ap.add_argument("--captions_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=512, help="square training image size")
    args = ap.parse_args()

    cat_dir = os.path.join(args.src_dir, args.category)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.captions_json) as f:
        data = json.load(f)
    captions = data["captions"][args.category]
    sources = data["sources"][args.category]

    records = []
    for i, (caption, source_rel) in enumerate(zip(captions, sources)):
        bad_path = resolve_source(cat_dir, source_rel)
        if bad_path is None:
            print(f"[skip] image not found: {source_rel}")
            continue
        mask_path = find_mask(bad_path)
        if not os.path.exists(mask_path):
            print(f"[skip] mask not found for: {bad_path}")
            continue
        img = Image.open(bad_path).convert("RGB")
        crop = crop_to_mask(img, mask_path).resize((args.size, args.size))
        name = f"{i:04d}.png"
        crop.save(os.path.join(args.out_dir, name))
        records.append({"file_name": name, "text": caption})

    with open(os.path.join(args.out_dir, "metadata.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} crops + metadata.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
