"""
Evaluate all test-split captions against the full gallery.

Encodes all test captions in batches, computes cosine similarity against
the cached gallery features, and reports Rank-1/5/10, mAP, and mINP.

Usage:
    python eval_all.py \
        --checkpoint logs/CUHK-PEDES/pretrained/best_real.pth \
        --config     logs/CUHK-PEDES/pretrained/configs.yaml \
        --cache      data/gallery_cache.pt \
        --batch_size 128
"""

import argparse
import json
import os.path as op

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.bases import tokenize
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer


# ---------------------------------------------------------------------------
# Dataset for batched caption encoding
# ---------------------------------------------------------------------------
class CaptionDataset(Dataset):
    def __init__(self, captions, pids, tokenizer, text_length=77):
        self.tokens = [
            tokenize(c, tokenizer=tokenizer, text_length=text_length)
            for c in tqdm(captions, desc="Tokenising", leave=False)
        ]
        self.pids = pids

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.pids[i], self.tokens[i]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(similarity, query_pids, gallery_pids, ks=(1, 5, 10)):
    """
    similarity : [Q, G] float tensor
    query_pids : [Q] int tensor
    gallery_pids: [G] int tensor
    Returns dict with R@k for each k, mAP, mINP.
    """
    Q, G = similarity.shape
    sorted_idx = torch.argsort(similarity, dim=1, descending=True)  # [Q, G]

    ranks = torch.zeros(Q, dtype=torch.long)
    ap_list, inp_list = [], []

    for i in range(Q):
        qpid = query_pids[i].item()
        ordered_pids = gallery_pids[sorted_idx[i]]  # [G]
        matches = (ordered_pids == qpid)            # [G] bool

        n_pos = matches.sum().item()
        if n_pos == 0:
            continue

        # Rank of first match (0-indexed)
        first_match = matches.nonzero(as_tuple=True)[0][0].item()
        ranks[i] = first_match

        # Average Precision
        match_positions = matches.nonzero(as_tuple=True)[0].float() + 1  # 1-indexed
        precision_at_k = torch.arange(1, n_pos + 1, dtype=torch.float) / match_positions
        ap = precision_at_k.mean().item()
        ap_list.append(ap)

        # mINP: precision at the last relevant match
        last_match = match_positions[-1].item()
        inp_list.append(n_pos / last_match)

    results = {}
    for k in ks:
        results[f"R@{k}"] = (ranks < k).float().mean().item() * 100
    results["mAP"]  = (sum(ap_list)  / len(ap_list))  * 100 if ap_list  else 0
    results["mINP"] = (sum(inp_list) / len(inp_list)) * 100 if inp_list else 0
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--cache",      default="data/gallery_cache.pt")
    parser.add_argument("--data_dir",   default="data/CUHK-PEDES")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split",      default="test",
                        choices=["test", "val", "all"])
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    # ---- device ----
    if args.device != "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "cpu" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # ---- model ----
    cfg = load_train_configs(args.config)
    cfg.training = False
    raw = torch.load(args.checkpoint, map_location="cpu")
    sd = raw.get("model", raw)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    num_classes = next((v.shape[0] for k, v in sd.items() if k == "classifier.weight"), 11003)
    del raw, sd

    model = build_model(cfg, num_classes=num_classes)
    Checkpointer(model).load(f=args.checkpoint)
    model.to(device).eval()
    text_length = getattr(cfg, "text_length", 77)
    print(f"Model loaded on {device}")

    # ---- gallery cache ----
    print(f"Loading gallery cache from {args.cache}")
    cache = torch.load(args.cache, map_location="cpu")
    gallery_feats = cache["feats"]          # [G, 512] L2-normed
    gallery_pids  = cache["pids"]           # [G]
    print(f"Gallery: {len(gallery_pids)} images")

    # ---- load captions ----
    annos = json.load(open(op.join(args.data_dir, "reid_raw.json")))
    if args.split == "all":
        entries = annos
    else:
        entries = [a for a in annos if a["split"] == args.split]

    captions, query_pids = [], []
    for a in entries:
        for cap in a["captions"]:
            captions.append(cap)
            query_pids.append(int(a["id"]))

    print(f"Queries ({args.split} split): {len(captions)} captions "
          f"from {len(set(query_pids))} unique persons")

    # ---- encode all captions in batches ----
    tokenizer = SimpleTokenizer()
    cap_dataset = CaptionDataset(captions, query_pids, tokenizer, text_length)
    loader = DataLoader(cap_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_feats, all_pids = [], []
    with torch.no_grad():
        for pids, tokens in tqdm(loader, desc="Encoding captions"):
            tokens = tokens.to(device)
            feats  = model.encode_text(tokens)
            feats  = F.normalize(feats, p=2, dim=1)
            all_feats.append(feats.cpu())
            all_pids.append(pids)

    query_feats = torch.cat(all_feats, dim=0)   # [Q, 512]
    query_pids  = torch.cat(all_pids,  dim=0)   # [Q]

    # ---- similarity & metrics ----
    print("Computing similarity matrix...")
    # Process in chunks to avoid OOM on large Q
    chunk = 2000
    sim_chunks = []
    for i in range(0, len(query_feats), chunk):
        sim_chunks.append(query_feats[i:i+chunk] @ gallery_feats.t())
    similarity = torch.cat(sim_chunks, dim=0)   # [Q, G]

    metrics = compute_metrics(similarity, query_pids, gallery_pids)

    # ---- print ----
    table = PrettyTable(["Metric", "Value"])
    for k, v in metrics.items():
        table.add_row([k, f"{v:.2f}%"])
    print(f"\nResults on {args.split} split ({len(captions)} queries, {len(gallery_pids)}-image gallery):")
    print(table)

    # ---- sample: show top-5 for first 5 queries ----
    print("\nSample results (first 5 queries):")
    img_paths = cache["paths"]
    for i in range(min(5, len(captions))):
        scores, indices = torch.topk(similarity[i], k=5)
        detail = PrettyTable(["Rank", "PID", "Score", "File"])
        for rank, (idx, sc) in enumerate(zip(indices.tolist(), scores.tolist()), 1):
            pid = gallery_pids[idx].item()
            detail.add_row([rank, pid, f"{sc:.4f}", op.basename(img_paths[idx])])
        print(f'\nQuery {i+1} (PID {query_pids[i].item()}): "{captions[i][:80]}"')
        print(detail)


if __name__ == "__main__":
    main()
