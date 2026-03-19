"""
Download CUHK-PEDES from HuggingFace and convert to IRRA's expected format.

Source: MaulikMadhavi/CUHK-PEDES-processed
  https://huggingface.co/datasets/MaulikMadhavi/CUHK-PEDES-processed/viewer

Output layout (what IRRA's CUHKPEDES loader expects):
  data/CUHK-PEDES/imgs/          All images (flat, no subdirectories)
  data/CUHK-PEDES/reid_raw.json  Annotations with train/val/test splits

Usage:
  # Download fresh from HuggingFace:
  python prepare_data.py

  # Reuse already-downloaded images (avoids re-downloading ~700 MB):
  python prepare_data.py --images-source ../text-image-reid/data/images

  # Custom output directory:
  python prepare_data.py --output-dir ./data
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as op
import re
import shutil
import sys
from collections import defaultdict

# The local datasets/ directory would shadow the HuggingFace 'datasets' package.
# Remove the project root from sys.path temporarily so HF datasets is importable.
_project_root = op.dirname(op.abspath(__file__))
_patched_path = [p for p in sys.path if op.abspath(p) != _project_root and p != '']
sys.path = _patched_path
from datasets import load_dataset as _hf_load_dataset  # HuggingFace datasets
sys.path.insert(0, _project_root)  # restore

from tqdm import tqdm


def _person_id(filename: str) -> str:
    """Extract person identity string from a CUHK-PEDES filename."""
    stem = op.splitext(filename)[0]
    # Format: p<id>_s<sample>  e.g. p8130_s10935
    m = re.match(r"(p\d+)_s\d+", stem)
    if m:
        return m.group(1)
    # Format: <4-digit person><3-digit index>  e.g. 0363004
    m = re.match(r"(\d{4})\d+", stem)
    if m:
        return m.group(1)
    return stem  # fallback


def _sort_key(pid_str: str):
    """Sort numeric IDs before alphanumeric ones."""
    m = re.match(r"^(\d+)$", pid_str)
    if m:
        return (0, int(m.group(1)))
    return (1, pid_str)


def main():
    parser = argparse.ArgumentParser(description="Prepare CUHK-PEDES for IRRA training.")
    parser.add_argument("--output-dir", default="./data",
                        help="Root output directory (default: ./data)")
    parser.add_argument("--images-source", default=None,
                        help="Path to existing images directory to reuse instead of downloading. "
                             "E.g. ../text-image-reid/data/images")
    args = parser.parse_args()

    imgs_dir = op.join(args.output_dir, "CUHK-PEDES", "imgs")
    anno_path = op.join(args.output_dir, "CUHK-PEDES", "reid_raw.json")
    os.makedirs(imgs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build person_id -> [{filename, captions}] mapping
    # ------------------------------------------------------------------
    persons: dict[str, list[dict]] = defaultdict(list)

    if args.images_source:
        # Reuse existing images + re-download captions from HuggingFace
        src = op.abspath(args.images_source)
        print(f"Reusing images from: {src}")
        print("Loading captions from HuggingFace (MaulikMadhavi/CUHK-PEDES-processed)...")
        ds = _hf_load_dataset("MaulikMadhavi/CUHK-PEDES-processed", split="train")
        print(f"Loaded {len(ds)} rows")

        for row in tqdm(ds, desc="Processing"):
            filename = row["filename"]
            pid_str = _person_id(filename)

            # Symlink or copy image from source
            dst = op.join(imgs_dir, filename)
            if not op.exists(dst):
                src_img = op.join(src, filename)
                if op.exists(src_img):
                    os.symlink(src_img, dst)
                else:
                    # Fall back to saving from the HuggingFace PIL image
                    row["image"].convert("RGB").save(dst)

            persons[pid_str].append({
                "filename": filename,
                "captions": row["text"],
            })
    else:
        # Full download from HuggingFace
        print("Loading MaulikMadhavi/CUHK-PEDES-processed from HuggingFace...")
        ds = _hf_load_dataset("MaulikMadhavi/CUHK-PEDES-processed", split="train")
        print(f"Loaded {len(ds)} rows")

        for row in tqdm(ds, desc="Downloading images"):
            filename = row["filename"]
            pid_str = _person_id(filename)

            dst = op.join(imgs_dir, filename)
            if not op.exists(dst):
                row["image"].convert("RGB").save(dst)

            persons[pid_str].append({
                "filename": filename,
                "captions": row["text"],
            })

    # ------------------------------------------------------------------
    # Create train / val / test splits by person identity
    # ------------------------------------------------------------------
    sorted_pids = sorted(persons.keys(), key=_sort_key)
    n = len(sorted_pids)

    # ~80% train, ~10% val, ~10% test  (mirrors CUHK-PEDES proportions)
    n_test = max(500, int(n * 0.08))
    n_val  = max(500, int(n * 0.08))
    n_train = n - n_test - n_val

    train_pids = set(sorted_pids[:n_train])
    val_pids   = set(sorted_pids[n_train : n_train + n_val])
    test_pids  = set(sorted_pids[n_train + n_val :])

    print(f"\nPerson ID split: {len(train_pids)} train | {len(val_pids)} val | {len(test_pids)} test")

    # Assign consecutive integer IDs (IRRA requires 1-indexed consecutive IDs)
    pid_to_int: dict[str, int] = {pid: i + 1 for i, pid in enumerate(sorted_pids)}

    # ------------------------------------------------------------------
    # Build reid_raw.json
    # ------------------------------------------------------------------
    annotations = []
    for pid_str in tqdm(sorted_pids, desc="Building reid_raw.json"):
        int_id = pid_to_int[pid_str]
        split = "train" if pid_str in train_pids else ("val" if pid_str in val_pids else "test")
        for img_info in persons[pid_str]:
            annotations.append({
                "split": split,
                "id": int_id,
                "file_path": img_info["filename"],  # relative to imgs/
                "captions": img_info["captions"],
            })

    with open(anno_path, "w") as f:
        json.dump(annotations, f)

    n_imgs = sum(len(v) for v in persons.values())
    n_captions = sum(len(img["captions"]) for imgs in persons.values() for img in imgs)
    print(f"\nDone!")
    print(f"  Images  : {imgs_dir}  ({n_imgs} files)")
    print(f"  Annots  : {anno_path}  ({len(annotations)} entries, {n_captions} captions)")
    print(f"\nNext step: bash run_train.sh")


if __name__ == "__main__":
    main()
