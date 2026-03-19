"""
Text-to-image person retrieval using a fine-tuned IRRA model.

Given a natural language query, encodes all gallery images, computes cosine
similarity, and returns the top-k most similar images.

Usage:
    python retrieve.py \
        --query "a woman in a red jacket" \
        --checkpoint logs/CUHK-PEDES/20240101_120000_irra/best.pth \
        --config    logs/CUHK-PEDES/20240101_120000_irra/configs.yaml \
        --gallery_dir data/CUHK-PEDES \
        --top_k 5 \
        --output_dir results/ \
        [--save_cache gallery_cache.pt] \
        [--load_cache gallery_cache.pt]
"""

import argparse
import json
import os
import os.path as op
import shutil

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.bases import ImageDataset, tokenize
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer


def encode_gallery(model, gallery_dir, img_size, device, batch_size=256, num_workers=4,
                   use_full_gallery=False):
    """Encode gallery images into L2-normalized feature vectors.

    Args:
        use_full_gallery: if True, encode all images across all splits (train+val+test).
                          if False (default), encode only the test split.
    """
    dataset = CUHKPEDES(root=gallery_dir)
    if use_full_gallery:
        # Collect unique (pid, path) from all splits
        imgs_dir = op.join(gallery_dir, 'CUHK-PEDES', 'imgs/')
        annos = json.load(open(op.join(gallery_dir, 'CUHK-PEDES', 'reid_raw.json')))
        seen = set()
        image_pids, img_paths = [], []
        for a in annos:
            p = op.join(imgs_dir, a['file_path'])
            if p not in seen:
                seen.add(p)
                image_pids.append(int(a['id']))
                img_paths.append(p)
    else:
        ds = dataset.test
        img_paths = ds['img_paths']
        image_pids = ds['image_pids']

    transform = build_transforms(img_size=img_size, is_train=False)
    gallery_set = ImageDataset(image_pids, img_paths, transform)
    loader = DataLoader(gallery_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    all_feats = []
    all_pids = []
    model.eval()
    with torch.no_grad():
        for pids, imgs in tqdm(loader, desc="Encoding gallery"):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            all_feats.append(feats.cpu())
            all_pids.append(pids)

    gallery_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    gallery_pids = torch.cat(all_pids, dim=0)
    return gallery_feats, gallery_pids, img_paths


def encode_query(model, query_text, device, text_length=77):
    """Tokenize and encode a text query into an L2-normalized feature vector."""
    tokenizer = SimpleTokenizer()
    tokens = tokenize(query_text, tokenizer=tokenizer, text_length=text_length)
    tokens = tokens.unsqueeze(0).to(device)  # [1, text_length]

    model.eval()
    with torch.no_grad():
        feat = model.encode_text(tokens)

    return F.normalize(feat, p=2, dim=1)  # [1, D]


def retrieve(args):
    # ------------------------------------------------------------------
    # Load training config and build model
    # ------------------------------------------------------------------
    cfg = load_train_configs(args.config)
    cfg.training = False

    if args.device != "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "cpu" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Detect num_classes from the checkpoint's classifier head shape so that
    # a pretrained checkpoint (e.g. trained on the original 11 003-person
    # CUHK-PEDES split) loads cleanly regardless of our local dataset size.
    _raw = torch.load(args.checkpoint, map_location='cpu')
    _sd = _raw.get('model', _raw)
    # strip DDP "module." prefix if present
    _sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in _sd.items()}
    num_classes = next(
        (v.shape[0] for k, v in _sd.items() if k == 'classifier.weight'),
        11003,  # fallback: original CUHK-PEDES
    )
    del _raw, _sd

    model = build_model(cfg, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=args.checkpoint)
    model.to(device)
    model.eval()

    img_size = tuple(cfg.img_size) if hasattr(cfg, 'img_size') else (384, 128)
    text_length = cfg.text_length if hasattr(cfg, 'text_length') else 77

    # ------------------------------------------------------------------
    # Gallery features (with optional caching)
    # ------------------------------------------------------------------
    if args.load_cache and op.exists(args.load_cache):
        print(f"Loading gallery cache from {args.load_cache}")
        cache = torch.load(args.load_cache, map_location='cpu')
        gallery_feats = cache['feats']
        gallery_pids = cache['pids']
        img_paths = cache['paths']
    else:
        gallery_feats, gallery_pids, img_paths = encode_gallery(
            model, args.gallery_dir, img_size, device,
            use_full_gallery=getattr(args, 'full_gallery', False))
        if args.save_cache:
            os.makedirs(op.dirname(op.abspath(args.save_cache)), exist_ok=True)
            torch.save({'feats': gallery_feats, 'pids': gallery_pids,
                        'paths': img_paths}, args.save_cache)
            print(f"Gallery cache saved to {args.save_cache}")

    # ------------------------------------------------------------------
    # Encode text query
    # ------------------------------------------------------------------
    query_feat = encode_query(model, args.query, device, text_length)  # [1, D]
    query_feat = query_feat.cpu()

    # ------------------------------------------------------------------
    # Cosine similarity and top-k ranking
    # ------------------------------------------------------------------
    similarity = query_feat @ gallery_feats.t()  # [1, N]
    scores, indices = torch.topk(similarity[0], k=args.top_k, largest=True, sorted=True)

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    table = PrettyTable(["Rank", "Person ID", "Score", "Image Path"])
    results = []
    for rank, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist()), start=1):
        pid = gallery_pids[idx].item()
        path = img_paths[idx]
        table.add_row([rank, pid, f"{score:.4f}", path])
        results.append({"rank": rank, "pid": pid, "score": score, "path": path})

    print(f"\nQuery: \"{args.query}\"")
    print(table)

    # ------------------------------------------------------------------
    # Copy top-k images to output_dir
    # ------------------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for item in results:
            ext = op.splitext(item['path'])[1]
            dst = op.join(args.output_dir, f"rank{item['rank']}_pid{item['pid']}{ext}")
            shutil.copy2(item['path'], dst)
        print(f"\nTop-{args.top_k} images saved to: {args.output_dir}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-image person retrieval")
    parser.add_argument("--query", required=True,
                        help="Natural language description of a person")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pth checkpoint file")
    parser.add_argument("--config", required=True,
                        help="Path to configs.yaml saved during training")
    parser.add_argument("--gallery_dir", default="data",
                        help="Root directory containing CUHK-PEDES/ (default: data)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results to return (default: 5)")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save top-k images (default: results)")
    parser.add_argument("--save_cache", default=None,
                        help="Path to save gallery feature cache (.pt)")
    parser.add_argument("--load_cache", default=None,
                        help="Path to load gallery feature cache (.pt)")
    parser.add_argument("--device", default="cuda",
                        help="Device to run inference on (default: cuda)")
    parser.add_argument("--full_gallery", action="store_true",
                        help="Use all images (train+val+test) as gallery instead of test split only")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    retrieve(args)
