"""
Encode all gallery images into feature vectors, saving progress after every batch
so the job can be interrupted and resumed without losing work.

Progress is tracked in <cache_dir>/encode_progress.json.
Partial batch features are saved as <cache_dir>/batch_NNNNN.pt files.
On completion these are merged into gallery_cache.pt and the partials are removed.

Usage:
    python encode_gallery.py \
        --checkpoint logs/CUHK-PEDES/pretrained/best_real.pth \
        --config    logs/CUHK-PEDES/pretrained/configs_orig.yaml \
        --gallery_dir data \
        --cache_dir  data/gallery_cache_parts \
        --output     data/gallery_cache.pt \
        --batch_size 64 \
        --full_gallery

Resume (just run the same command again):
    python encode_gallery.py ...  # picks up where it left off

Check status:
    python encode_gallery.py --status --cache_dir data/gallery_cache_parts
"""

import argparse
import json
import os
import os.path as op
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.bases import ImageDataset
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _progress_path(cache_dir):
    return op.join(cache_dir, 'encode_progress.json')


def _batch_path(cache_dir, batch_idx):
    return op.join(cache_dir, f'batch_{batch_idx:06d}.pt')


def load_progress(cache_dir):
    p = _progress_path(cache_dir)
    if op.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_progress(cache_dir, prog):
    with open(_progress_path(cache_dir), 'w') as f:
        json.dump(prog, f, indent=2)


def build_model_from_args(args):
    cfg = load_train_configs(args.config)
    cfg.training = False

    if args.device != "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "cpu" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    raw = torch.load(args.checkpoint, map_location='cpu')
    sd = raw.get('model', raw)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    num_classes = next((v.shape[0] for k, v in sd.items() if k == 'classifier.weight'), 11003)
    del raw, sd

    model = build_model(cfg, num_classes=num_classes)
    Checkpointer(model).load(f=args.checkpoint)
    model.to(device).eval()
    print(f"Model loaded on {device} (num_classes={num_classes})")
    return model, cfg, device


def collect_all_images(gallery_dir, full_gallery):
    """Return (image_pids, img_paths) for the selected gallery split."""
    if full_gallery:
        imgs_dir = op.join(gallery_dir, 'CUHK-PEDES', 'imgs/')
        with open(op.join(gallery_dir, 'CUHK-PEDES', 'reid_raw.json')) as f:
            annos = json.load(f)
        seen = set()
        pids, paths = [], []
        for a in annos:
            p = op.join(imgs_dir, a['file_path'])
            if p not in seen:
                seen.add(p)
                pids.append(int(a['id']))
                paths.append(p)
    else:
        ds = CUHKPEDES(root=gallery_dir, verbose=False)
        paths = ds.test['img_paths']
        pids = ds.test['image_pids']
    return pids, paths


# ------------------------------------------------------------------ #
# Encoding
# ------------------------------------------------------------------ #

def encode(args):
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load or init progress
    prog = load_progress(args.cache_dir)
    if prog is not None:
        print(f"Resuming from batch {prog['next_batch']} / {prog['total_batches']} "
              f"({prog['images_done']} / {prog['total_images']} images done)")
        img_paths = prog['img_paths']
        image_pids = prog['image_pids']
    else:
        print("Starting fresh encoding run.")
        image_pids, img_paths = collect_all_images(args.gallery_dir, args.full_gallery)
        total = len(img_paths)
        n_batches = (total + args.batch_size - 1) // args.batch_size
        prog = {
            'checkpoint': op.abspath(args.checkpoint),
            'config': op.abspath(args.config),
            'total_images': total,
            'total_batches': n_batches,
            'next_batch': 0,
            'images_done': 0,
            'img_paths': img_paths,
            'image_pids': image_pids,
        }
        save_progress(args.cache_dir, prog)
        print(f"Gallery: {total} images → {n_batches} batches of {args.batch_size}")

    if prog['next_batch'] >= prog['total_batches']:
        print("All batches already encoded. Merging to final cache.")
        _merge(args, prog)
        return

    # Build model only if there's actual work to do
    model, cfg, device = build_model_from_args(args)
    img_size = tuple(cfg.img_size) if hasattr(cfg, 'img_size') else (384, 128)
    transform = build_transforms(img_size=img_size, is_train=False)

    all_pids = prog['image_pids']
    all_paths = prog['img_paths']
    total = prog['total_images']
    bs = args.batch_size
    n_batches = prog['total_batches']

    start_batch = prog['next_batch']
    pbar = tqdm(range(start_batch, n_batches), desc="Encoding batches",
                initial=start_batch, total=n_batches)

    for b in pbar:
        start = b * bs
        end = min(start + bs, total)
        batch_pids = all_pids[start:end]
        batch_paths = all_paths[start:end]

        dataset = ImageDataset(batch_pids, batch_paths, transform)
        loader = DataLoader(dataset, batch_size=len(batch_paths), shuffle=False,
                            num_workers=args.num_workers)

        with torch.no_grad():
            for pids_t, imgs in loader:
                imgs = imgs.to(device)
                feats = model.encode_image(imgs)
                feats = F.normalize(feats.cpu(), p=2, dim=1)

        torch.save({'feats': feats, 'pids': pids_t}, _batch_path(args.cache_dir, b))

        prog['next_batch'] = b + 1
        prog['images_done'] = end
        save_progress(args.cache_dir, prog)
        pbar.set_postfix(images=end)

    print("\nAll batches encoded. Merging...")
    _merge(args, prog)


def _merge(args, prog):
    """Merge all batch_*.pt files into a single gallery_cache.pt."""
    n_batches = prog['total_batches']
    all_feats, all_pids = [], []

    missing = []
    for b in range(n_batches):
        bp = _batch_path(args.cache_dir, b)
        if not op.exists(bp):
            missing.append(b)

    if missing:
        print(f"WARNING: {len(missing)} batch files missing: {missing[:10]}...")
        print("Re-run encode_gallery.py to fill the gaps.")
        return

    print(f"Merging {n_batches} batch files...")
    for b in tqdm(range(n_batches), desc="Merging"):
        d = torch.load(_batch_path(args.cache_dir, b), map_location='cpu')
        all_feats.append(d['feats'])
        all_pids.append(d['pids'])

    gallery_feats = torch.cat(all_feats, dim=0)
    gallery_pids = torch.cat(all_pids, dim=0)
    img_paths = prog['img_paths']

    os.makedirs(op.dirname(op.abspath(args.output)), exist_ok=True)
    torch.save({'feats': gallery_feats, 'pids': gallery_pids, 'paths': img_paths}, args.output)
    print(f"Saved gallery cache: {args.output}  ({len(img_paths)} images, shape {gallery_feats.shape})")

    # Clean up partial files
    if not args.keep_parts:
        for b in range(n_batches):
            os.remove(_batch_path(args.cache_dir, b))
        os.remove(_progress_path(args.cache_dir))
        try:
            os.rmdir(args.cache_dir)
        except OSError:
            pass  # not empty, leave it
        print("Cleaned up partial batch files.")


def status(args):
    prog = load_progress(args.cache_dir)
    if prog is None:
        print("No encoding progress found in", args.cache_dir)
        return
    pct = 100.0 * prog['images_done'] / max(prog['total_images'], 1)
    print(f"Progress: batch {prog['next_batch']} / {prog['total_batches']}  "
          f"({prog['images_done']} / {prog['total_images']} images, {pct:.1f}%)")
    # Count saved batch files
    saved = sum(1 for b in range(prog['total_batches']) if op.exists(_batch_path(args.cache_dir, b)))
    print(f"Batch files on disk: {saved} / {prog['total_batches']}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Encode gallery images with resumable progress")
    p.add_argument("--checkpoint", default="logs/CUHK-PEDES/pretrained/best_real.pth")
    p.add_argument("--config",     default="logs/CUHK-PEDES/pretrained/configs_orig.yaml")
    p.add_argument("--gallery_dir",default="data")
    p.add_argument("--cache_dir",  default="data/gallery_cache_parts",
                   help="Directory for partial batch files and progress state")
    p.add_argument("--output",     default="data/gallery_cache.pt",
                   help="Final merged cache file")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--full_gallery", action="store_true",
                   help="Use all 34k images (train+val+test) instead of just test split")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--keep_parts", action="store_true",
                   help="Keep partial batch files after merging (default: delete)")
    p.add_argument("--status",     action="store_true",
                   help="Print current progress and exit (no encoding)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.status:
        status(args)
    else:
        encode(args)
