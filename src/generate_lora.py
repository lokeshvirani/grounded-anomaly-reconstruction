"""Generate defect crops with SD 1.5, optionally with a trained LoRA.

Run it twice for the before/after comparison:
  - without --lora  -> base SD 1.5            (the "before")
  - with    --lora  -> SD 1.5 + your adapter  (the "after")
"""
import argparse
import json
import os

import torch
from diffusers import StableDiffusionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--lora", default=None, help="path to LoRA weights dir (omit = base model)")
    ap.add_argument("--captions_json", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_caption", type=int, default=4, help="images generated per caption")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype,
                                                   safety_checker=None)
    pipe = pipe.to(device)
    if args.lora:                              # load the trained adapter (the "after")
        pipe.load_lora_weights(args.lora)
    pipe.set_progress_bar_config(disable=True)

    with open(args.captions_json) as f:
        captions = json.load(f)["captions"][args.category]

    os.makedirs(args.out_dir, exist_ok=True)
    idx = 0
    for caption in captions:
        for _ in range(args.per_caption):
            gen = torch.Generator(device=device).manual_seed(args.seed + idx)
            img = pipe(caption, num_inference_steps=args.steps,
                       height=args.size, width=args.size, generator=gen).images[0]
            img.resize((256, 256)).save(os.path.join(args.out_dir, f"{idx:04d}.png"))
            idx += 1
        print(f"[{idx}] {caption[:60]}")
    print(f"wrote {idx} images to {args.out_dir}")


if __name__ == "__main__":
    main()
