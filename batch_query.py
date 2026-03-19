"""
Run a batch of text queries against the gallery and report retrieval results.

Usage:
    python batch_query.py \
        --checkpoint logs/CUHK-PEDES/pretrained/best_real.pth \
        --config    logs/CUHK-PEDES/pretrained/configs_orig.yaml \
        --gallery_dir data \
        --cache     data/gallery_cache.pt \
        --n_queries 100 \
        --top_k 5
"""

import argparse
import json
import os
import os.path as op
import random
import types

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

from retrieve import encode_gallery, encode_query, retrieve
from utils.iotools import load_train_configs
from model import build_model
from utils.checkpoint import Checkpointer


SAMPLE_QUERIES = [
    "a woman wearing a red jacket and black trousers",
    "a man in a blue shirt and dark jeans",
    "a person in a yellow dress carrying a bag",
    "an old man with white hair wearing a grey coat",
    "a young woman with long hair in a white shirt",
    "a man wearing a black hoodie and sports shoes",
    "a woman in a floral dress with a handbag",
    "a tall man in a dark suit",
    "a child wearing a green t-shirt and shorts",
    "a woman in a pink cardigan and blue jeans",
    "a man with a backpack wearing a striped shirt",
    "a woman in a long skirt and boots",
    "a man in a red polo shirt and khaki pants",
    "a person wearing a white hoodie and grey sweatpants",
    "a woman with short hair in a black blazer",
    "a man wearing glasses and a button-down shirt",
    "a woman carrying a large red bag",
    "a person in a navy blue jacket and sneakers",
    "a man with a beard wearing a brown jacket",
    "a woman in athletic wear with running shoes",
    "a person wearing a winter coat and scarf",
    "a man in a denim jacket and black trousers",
    "a woman with curly hair in a green top",
    "a person carrying a yellow umbrella",
    "a man in a white t-shirt and cargo pants",
    "a woman wearing a leopard print top",
    "a person with a laptop bag in formal clothes",
    "a man in a plaid shirt and jeans",
    "a woman in a black dress and heels",
    "a person in orange work clothes",
    "a man wearing a flat cap and overcoat",
    "a woman with a ponytail in a purple top",
    "a person in military-style clothing",
    "a man with a mustache in a formal suit",
    "a woman wearing a beige trench coat",
    "a person in a school uniform",
    "a man with tattoos in a sleeveless shirt",
    "a woman in a light blue summer dress",
    "a person wearing a cap and a dark jacket",
    "a man carrying a briefcase in work attire",
    "a woman in a long red coat and boots",
    "a person in workout clothes with earphones",
    "a man wearing a white polo shirt",
    "a woman with glasses in a formal outfit",
    "a person in casual clothes with a baseball cap",
    "a man with grey hair wearing a checked jacket",
    "a woman in skinny jeans and a crop top",
    "a person in a bright orange shirt",
    "a man wearing a dark turtleneck sweater",
    "a woman carrying a tote bag in business casual",
    "a person in a tracksuit and sneakers",
    "a man in a leather jacket and jeans",
    "a woman with a bun hairstyle in office clothes",
    "a person carrying a shopping bag",
    "a man in a blue windbreaker",
    "a woman in a midi skirt and blouse",
    "a person with a helmet and safety vest",
    "a man in casual summer wear with flip flops",
    "a woman in a sequined top and dark pants",
    "a person wearing a knit sweater and scarf",
    "a man in dark clothing carrying a duffel bag",
    "a woman in a white lace blouse and black skirt",
    "a person wearing shorts and a t-shirt",
    "a man with a backpack and headphones",
    "a woman in a kimono-style top",
    "a person in a high-visibility jacket",
    "a man in slacks and a polo shirt",
    "a woman with braided hair in a sundress",
    "a person in a rain jacket with an umbrella",
    "a man wearing a flat iron and vest",
    "a woman in a fur-trimmed coat",
    "a person in a school blazer with a tie",
    "a man in camo pants and a plain t-shirt",
    "a woman in high heels and a pencil skirt",
    "a person carrying a camera bag",
    "a man in a jogging suit doing exercise",
    "a woman in a flared skirt and wedge shoes",
    "a person wearing a bomber jacket",
    "a man in overalls and a work shirt",
    "a woman in a long floral maxi dress",
    "a person with a straw hat and light clothes",
    "a man in a polo neck and chinos",
    "a woman in a velvet blazer and silk blouse",
    "a person on a bicycle wearing a helmet",
    "a man carrying grocery bags",
    "a woman in yoga pants and a sports bra",
    "a person in a traditional ethnic costume",
    "a man wearing a beanie and puffer jacket",
    "a woman in a wrap dress and sandals",
    "a person in smart casual with loafers",
    "a man in dark blue jeans and a white henley",
    "a woman wearing a wide-brim hat and blouse",
    "a person in a mechanic uniform",
    "a man in a peacoat and scarf",
    "a woman in a pleated skirt and cardigan",
    "a person carrying a sports equipment bag",
    "a man in shorts and a polo at the beach",
    "a woman in a velvet dress and pearls",
]


def main():
    parser = argparse.ArgumentParser(description="Batch text-to-image person retrieval queries")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gallery_dir", default="data")
    parser.add_argument("--cache", default="data/gallery_cache.pt",
                        help="Path to gallery cache (.pt); created if missing")
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--full_gallery", action="store_true",
                        help="Use all 34k images instead of just test split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
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
    checkpointer = Checkpointer(model)
    checkpointer.load(f=args.checkpoint)
    model.to(device).eval()
    print(f"Model loaded on {device} (num_classes={num_classes})")

    img_size = tuple(cfg.img_size) if hasattr(cfg, 'img_size') else (384, 128)
    text_length = cfg.text_length if hasattr(cfg, 'text_length') else 77

    # ------------------------------------------------------------------ #
    # Gallery embeddings (cache)
    # ------------------------------------------------------------------ #
    if args.cache and op.exists(args.cache):
        print(f"Loading gallery cache from {args.cache}")
        cache = torch.load(args.cache, map_location='cpu')
        gallery_feats = cache['feats']
        gallery_pids = cache['pids']
        img_paths = cache['paths']
    else:
        gallery_feats, gallery_pids, img_paths = encode_gallery(
            model, args.gallery_dir, img_size, device,
            use_full_gallery=args.full_gallery)
        if args.cache:
            os.makedirs(op.dirname(op.abspath(args.cache)), exist_ok=True)
            torch.save({'feats': gallery_feats, 'pids': gallery_pids,
                        'paths': img_paths}, args.cache)
            print(f"Gallery cache saved to {args.cache}")

    print(f"\nGallery: {len(img_paths)} images")

    # ------------------------------------------------------------------ #
    # Run queries
    # ------------------------------------------------------------------ #
    queries = SAMPLE_QUERIES[:args.n_queries]
    if args.n_queries > len(SAMPLE_QUERIES):
        print(f"Only {len(SAMPLE_QUERIES)} sample queries available; using all of them.")
        queries = SAMPLE_QUERIES

    print(f"\nRunning {len(queries)} queries (top-{args.top_k})\n{'='*70}")
    summary = PrettyTable(["#", "Query (truncated)", f"Rank-1 PID", "Rank-1 Score"])

    for i, q in enumerate(queries, 1):
        query_feat = encode_query(model, q, device, text_length).cpu()
        sim = query_feat @ gallery_feats.t()
        scores, indices = torch.topk(sim[0], k=args.top_k, largest=True, sorted=True)

        top1_idx = indices[0].item()
        top1_pid = gallery_pids[top1_idx].item() if hasattr(gallery_pids[top1_idx], 'item') else gallery_pids[top1_idx]
        top1_score = scores[0].item()
        top1_path = op.basename(img_paths[top1_idx])

        summary.add_row([i, q[:55], top1_pid, f"{top1_score:.4f}"])

        # Detailed table every 10 queries
        if i <= 5 or i % 10 == 0:
            detail = PrettyTable(["Rank", "PID", "Score", "File"])
            for rank, (idx, sc) in enumerate(zip(indices.tolist(), scores.tolist()), 1):
                pid = gallery_pids[idx].item() if hasattr(gallery_pids[idx], 'item') else gallery_pids[idx]
                detail.add_row([rank, pid, f"{sc:.4f}", op.basename(img_paths[idx])])
            print(f"\nQuery {i}: \"{q}\"")
            print(detail)

    print(f"\n{'='*70}\nSummary of all {len(queries)} queries:")
    print(summary)


if __name__ == '__main__':
    main()
