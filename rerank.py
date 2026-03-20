"""
MLLM-reranked text-to-image person retrieval.

Runs IRRA retrieval to get top-k candidates, then asks Qwen2-VL (via
HuggingFace transformers) to rerank all k images against the text query in a
single comparative prompt.

Usage:
    python rerank.py \\
        --query "a woman in a red jacket" \\
        --checkpoint logs/.../best.pth \\
        --config    logs/.../configs.yaml \\
        --rerank_model Qwen/Qwen2-VL-7B-Instruct \\
        [--top_k 10] \\
        [--alpha 0.5] \\
        [--gallery_dir data] \\
        [--output_dir results/reranked] \\
        [--load_cache gallery_cache.pt]
"""

import argparse
import os
import os.path as op
import re
import shutil

import numpy as np
import torch
from prettytable import PrettyTable
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from retrieve import retrieve


def _build_prompt(query, k):
    lines = [
        f"I have {k} candidate images retrieved for a text query describing a person.",
        f'Query: "{query}"',
        "",
        f"Image 1 through Image {k} are shown above in order.",
        "",
        "Rerank these images from most to least likely to match the person described in the query.",
        f'Reply with only the image numbers in order, comma-separated (e.g. "3, 1, 4, 2, 5").',
    ]
    return "\n".join(lines)


def _parse_ranking(response, k):
    """Extract a ranked list of 1-based indices from the model response.

    Any indices missing from the response are appended at the end in their
    original order so the output always covers all k candidates.
    """
    nums = re.findall(r'\b(\d+)\b', response)
    seen = set()
    order = []
    for n in nums:
        idx = int(n)
        if 1 <= idx <= k and idx not in seen:
            seen.add(idx)
            order.append(idx)
    for i in range(1, k + 1):
        if i not in seen:
            order.append(i)
    return order  # 1-based, most-relevant first


def _load_mllm(model_dir):
    """Load Qwen2-VL model and processor onto the best available device."""
    print(f"\nLoading MLLM from {model_dir} ...")
    if torch.cuda.is_available():
        device_map = "cuda"
    elif torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    print(f"MLLM loaded on {device_map}")
    return model, processor


def _run_mllm(message, model, processor):
    """Run a single chat message through Qwen2-VL; return the response string."""
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def mllm_rerank(results, query, model_dir):
    """Send all top-k images + query to Qwen2-VL in one prompt.

    Returns:
        mllm_scores  -- np.ndarray [k], 1.0 for model's top pick, 0.0 for last
        irra_norm    -- np.ndarray [k], IRRA cosine scores normalised to [0,1]
        raw_response -- str, raw model output for debugging
    """
    k = len(results)

    model, processor = _load_mllm(model_dir)

    # Build message: all images first, then the text instruction
    content = []
    for item in results:
        content.append({
            "type": "image",
            "image": item["path"],
            "min_pixels": 50176,
            "max_pixels": 50176,
        })
    content.append({"type": "text", "text": _build_prompt(query, k)})

    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    raw_response = _run_mllm(message, model, processor)
    print(f"MLLM response: {raw_response}")

    mllm_order = _parse_ranking(raw_response, k)  # 1-based, best first

    # Rank-1 → score 1.0, rank-k → score 0.0
    mllm_scores = np.zeros(k)
    for new_rank, orig_idx in enumerate(mllm_order):
        mllm_scores[orig_idx - 1] = (k - new_rank) / max(k - 1, 1)

    # Normalise IRRA cosine scores to [0, 1]
    irra_scores = np.array([r["score"] for r in results])
    lo, hi = irra_scores.min(), irra_scores.max()
    irra_norm = (irra_scores - lo) / (hi - lo) if hi > lo else np.ones(k)

    return mllm_scores, irra_norm, raw_response


def main():
    parser = argparse.ArgumentParser(description="MLLM-reranked text-to-image person retrieval")

    # retrieve.py args (forwarded as-is)
    parser.add_argument("--query", required=True,
                        help="Natural language description of a person")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pth checkpoint file")
    parser.add_argument("--config", required=True,
                        help="Path to configs.yaml saved during training")
    parser.add_argument("--gallery_dir", default="data",
                        help="Root directory containing CUHK-PEDES/ (default: data)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of IRRA candidates to retrieve and rerank (default: 10)")
    parser.add_argument("--save_cache", default=None)
    parser.add_argument("--load_cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full_gallery", action="store_true")

    # reranking args
    parser.add_argument("--rerank_model", required=True,
                        help="HuggingFace model ID or local path (e.g. Qwen/Qwen2-VL-7B-Instruct)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight given to IRRA score vs MLLM score (default: 0.5)")
    parser.add_argument("--output_dir", default="results/reranked",
                        help="Directory to save reranked images (default: results/reranked)")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: IRRA retrieval                                               #
    # ------------------------------------------------------------------ #
    retrieve_args = argparse.Namespace(
        query=args.query,
        checkpoint=args.checkpoint,
        config=args.config,
        gallery_dir=args.gallery_dir,
        top_k=args.top_k,
        output_dir=None,        # skip file copy inside retrieve()
        save_cache=args.save_cache,
        load_cache=args.load_cache,
        device=args.device,
        full_gallery=args.full_gallery,
    )
    results = retrieve(retrieve_args)   # [{"rank", "pid", "score", "path"}, ...]

    # ------------------------------------------------------------------ #
    # Step 2: MLLM reranking                                               #
    # ------------------------------------------------------------------ #
    mllm_scores, irra_norm, _ = mllm_rerank(results, args.query, args.rerank_model)

    # ------------------------------------------------------------------ #
    # Step 3: Combine and sort                                             #
    # ------------------------------------------------------------------ #
    combined = args.alpha * irra_norm + (1.0 - args.alpha) * mllm_scores
    new_order = np.argsort(-combined)   # descending

    # ------------------------------------------------------------------ #
    # Step 4: Print re-ranked table                                        #
    # ------------------------------------------------------------------ #
    table = PrettyTable(["New Rank", "Old Rank", "PID", "IRRA", "MLLM", "Combined", "Image Path"])
    reranked = []
    for new_rank, orig_idx in enumerate(new_order, start=1):
        item = results[orig_idx]
        table.add_row([
            new_rank,
            item["rank"],
            item["pid"],
            f"{irra_norm[orig_idx]:.3f}",
            f"{mllm_scores[orig_idx]:.3f}",
            f"{combined[orig_idx]:.3f}",
            item["path"],
        ])
        reranked.append({
            **item,
            "rank": new_rank,
            "irra_score": float(irra_norm[orig_idx]),
            "mllm_score": float(mllm_scores[orig_idx]),
            "combined_score": float(combined[orig_idx]),
        })

    print(f"\nQuery: \"{args.query}\"")
    print(table)

    # ------------------------------------------------------------------ #
    # Step 5: Save reranked images                                         #
    # ------------------------------------------------------------------ #
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for item in reranked:
            ext = op.splitext(item["path"])[1]
            dst = op.join(args.output_dir, f"rank{item['rank']}_pid{item['pid']}{ext}")
            shutil.copy2(item["path"], dst)
        print(f"\nTop-{args.top_k} reranked images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
