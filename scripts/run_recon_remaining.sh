#!/bin/bash
# Run grounded reconstruction for the remaining 5 categories across 2 GPUs.
cd /media/data/Lokesh/thesis-master-datascience
P=./thesis_env/bin/python

recon() {  # $1=gpu  $2=category  $3=captions_json
  CUDA_VISIBLE_DEVICES=$1 $P src/reconstruct_real_defects.py \
    --src_dir data/preprocessed --category "$2" --captions_json "$3" \
    --out_dirname recon_llmcaption_full --imgs_per_caption 4 --low_vram
}

( recon 0 fabric      llm_captions_fabric_full.json
  recon 0 sheet_metal llm_captions_sheet_metal_full.json
  recon 0 vial        llm_captions_vial_full.json ) > recon_gpu0.log 2>&1 &

( recon 1 rice        llm_captions_rice_full.json
  recon 1 wallplugs   llm_captions_wallplugs_full.json ) > recon_gpu1.log 2>&1 &

wait
echo "ALL RECON DONE"
