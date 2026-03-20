"""
Compare IRRA baseline vs MLLM-reranked retrieval on a random query subset.

For each sampled query, the top-k IRRA candidates are sent to Qwen2-VL in a
single comparative prompt. The model returns a ranked ordering; we patch the
similarity row by redistributing the top-k IRRA scores into that order, then
compute R@1/R@5/R@10/mAP/mINP on both the original and patched matrices.

Usage:
    python eval_rerank.py \\
        --checkpoint logs/.../best.pth \\
        --config     logs/.../configs.yaml \\
        --cache      data/gallery_cache.pt \\
        --rerank_model Qwen/Qwen2-VL-7B-Instruct \\
        [--num_queries 100] \\
        [--top_k 10] \\
        [--data_dir data/CUHK-PEDES] \\
        [--split test] \\
        [--seed 42]
"""

import argparse
import json
import os.path as op
import random

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from datasets.bases import tokenize
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer

# Reuse dataset, metric, and prompt helpers from existing scripts
from eval_all import CaptionDataset, compute_metrics
from rerank import _build_prompt, _parse_ranking, _load_mllm, _run_mllm


def _mllm_rank_query(query_text, img_paths, model, processor):
    """Send one query + its top-k images to the MLLM; return a 1-based ranked list."""
    k = len(img_paths)
    content = []
    for path in img_paths:
        content.append({
            "type": "image",
            "image": path,
            "min_pixels": 50176,
            "max_pixels": 50176,
        })
    content.append({"type": "text", "text": _build_prompt(query_text, k)})

    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": content},
    ]

    response = _run_mllm(message, model, processor)
    return _parse_ranking(response, k), response  # 1-based, best first


def patch_sim_row(sim_row, top_indices, mllm_order):
    """Redistribute top-k IRRA scores into the MLLM-specified order.

    top_indices : 1-D tensor of k gallery indices, already sorted descending by IRRA score
    mllm_order  : list of 1-based positions (best → worst), length k
    Returns a cloned sim_row with the top-k scores reassigned.
    """
    k          = len(top_indices)
    top_scores = sim_row[top_indices].clone()   # s_1 >= s_2 >= ... >= s_k
    patched    = sim_row.clone()
    for new_pos, orig_1based in enumerate(mllm_order):
        patched[top_indices[orig_1based - 1]] = top_scores[new_pos]
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs MLLM-reranked evaluation on a random query subset"
    )
    parser.add_argument("--checkpoint",          required=True)
    parser.add_argument("--config",              required=True)
    parser.add_argument("--cache",               default="data/gallery_cache.pt")
    parser.add_argument("--rerank_model",        required=True,
                        help="HuggingFace model ID or local path for Qwen2-VL")
    parser.add_argument("--num_queries",         type=int, default=100,
                        help="Number of test queries to evaluate (default: 100)")
    parser.add_argument("--top_k",               type=int, default=10,
                        help="Candidates per query sent to the MLLM (default: 10)")
    parser.add_argument("--data_dir",            default="data/CUHK-PEDES")
    parser.add_argument("--split",               default="test",
                        choices=["test", "val", "all"])
    parser.add_argument("--device",              default="cuda")
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--progress_file",       default=None,
                        help="JSON file to save/resume MLLM rankings (default: auto-named)")
    args = parser.parse_args()

    if args.progress_file is None:
        args.progress_file = f"rerank_progress_n{args.num_queries}_k{args.top_k}_seed{args.seed}.json"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #
    if args.device != "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "cpu" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # ------------------------------------------------------------------ #
    # IRRA model                                                           #
    # ------------------------------------------------------------------ #
    cfg = load_train_configs(args.config)
    cfg.training = False
    raw = torch.load(args.checkpoint, map_location="cpu")
    sd  = raw.get("model", raw)
    sd  = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    num_classes = next((v.shape[0] for k, v in sd.items() if k == "classifier.weight"), 11003)
    del raw, sd

    model = build_model(cfg, num_classes=num_classes)
    Checkpointer(model).load(f=args.checkpoint)
    model.to(device).eval()
    text_length = getattr(cfg, "text_length", 77)
    print(f"IRRA model loaded on {device}")

    # ------------------------------------------------------------------ #
    # Gallery cache                                                        #
    # ------------------------------------------------------------------ #
    print(f"Loading gallery cache from {args.cache}")
    cache        = torch.load(args.cache, map_location="cpu")
    gallery_feats = cache["feats"]   # [G, 512]
    gallery_pids  = cache["pids"]    # [G]
    img_paths     = cache["paths"]   # list[str]
    G = len(gallery_pids)
    print(f"Gallery: {G} images")

    # ------------------------------------------------------------------ #
    # Load & sample captions                                               #
    # ------------------------------------------------------------------ #
    annos = json.load(open(op.join(args.data_dir, "reid_raw.json")))
    entries = annos if args.split == "all" else [a for a in annos if a["split"] == args.split]

    all_captions, all_qpids = [], []
    for a in entries:
        for cap in a["captions"]:
            all_captions.append(cap)
            all_qpids.append(int(a["id"]))

    total = len(all_captions)
    n = min(args.num_queries, total)
    indices = random.sample(range(total), n)
    captions  = [all_captions[i] for i in indices]
    qpids_raw = [all_qpids[i]    for i in indices]
    print(f"Sampled {n}/{total} queries (seed={args.seed})")

    # ------------------------------------------------------------------ #
    # Encode sampled captions                                              #
    # ------------------------------------------------------------------ #
    tokenizer   = SimpleTokenizer()
    cap_dataset = CaptionDataset(captions, qpids_raw, tokenizer, text_length)
    loader      = DataLoader(cap_dataset, batch_size=128, shuffle=False, num_workers=0)

    all_feats, all_pids = [], []
    with torch.no_grad():
        for pids, tokens in tqdm(loader, desc="Encoding captions"):
            feats = F.normalize(model.encode_text(tokens.to(device)), p=2, dim=1)
            all_feats.append(feats.cpu())
            all_pids.append(pids)

    query_feats = torch.cat(all_feats, dim=0)   # [Q, 512]
    query_pids  = torch.cat(all_pids,  dim=0)   # [Q]

    # ------------------------------------------------------------------ #
    # Baseline similarity matrix [Q, G]                                   #
    # ------------------------------------------------------------------ #
    print("Computing baseline similarity matrix...")
    similarity_base = query_feats @ gallery_feats.t()   # [Q, G]

    # ------------------------------------------------------------------ #
    # Load progress cache (resume support)                                #
    # ------------------------------------------------------------------ #
    if op.exists(args.progress_file):
        with open(args.progress_file) as f:
            progress = json.load(f)
        print(f"Resuming from {args.progress_file} ({len(progress)}/{n} done)")
    else:
        progress = {}

    todo = [i for i in range(n) if str(i) not in progress]
    print(f"{len(todo)} queries remaining")

    # ------------------------------------------------------------------ #
    # Load MLLM (once, only if needed)                                    #
    # ------------------------------------------------------------------ #
    if todo:
        mllm_model, processor = _load_mllm(args.rerank_model)
    else:
        mllm_model = processor = None

    # ------------------------------------------------------------------ #
    # Reranking loop                                                       #
    # ------------------------------------------------------------------ #
    similarity_reranked = similarity_base.clone()

    for i in tqdm(todo, desc="Reranking queries"):
        # Top-k gallery indices for this query (sorted descending)
        _, top_indices = torch.topk(similarity_base[i], k=args.top_k, largest=True, sorted=True)
        top_img_paths  = [img_paths[idx] for idx in top_indices.tolist()]

        mllm_order, response = _mllm_rank_query(captions[i], top_img_paths, mllm_model, processor)

        progress[str(i)] = mllm_order
        with open(args.progress_file, "w") as f:
            json.dump(progress, f)

    # Apply all cached rankings (covers both freshly computed and resumed ones)
    for idx_str, mllm_order in progress.items():
        i = int(idx_str)
        _, top_indices = torch.topk(similarity_base[i], k=args.top_k, largest=True, sorted=True)
        similarity_reranked[i] = patch_sim_row(similarity_base[i], top_indices, mllm_order)

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    metrics_base     = compute_metrics(similarity_base,     query_pids, gallery_pids)
    metrics_reranked = compute_metrics(similarity_reranked, query_pids, gallery_pids)

    # ------------------------------------------------------------------ #
    # Print comparison table                                               #
    # ------------------------------------------------------------------ #
    print(f"\nEvaluating {n} queries (seed={args.seed}) against full gallery ({G} images)")
    print(f"MLLM: {args.rerank_model}  top_k={args.top_k}\n")

    keys = ["R@1", "R@5", "R@10", "mAP", "mINP"]
    table = PrettyTable([""] + keys)
    table.add_row(
        ["Baseline"] + [f"{metrics_base[k]:.2f}" for k in keys]
    )
    table.add_row(
        ["Reranked"] + [f"{metrics_reranked[k]:.2f}" for k in keys]
    )
    delta_row = ["Delta"]
    for k in keys:
        d = metrics_reranked[k] - metrics_base[k]
        delta_row.append(f"{d:+.2f}")
    table.add_row(delta_row)
    print(table)


if __name__ == "__main__":
    main()
