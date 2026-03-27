"""
Visual reranking on CUHK-PEDES test set.

For each query, sends the top-N IRRA candidates (as images) together with the
text query to Qwen2-VL in a single multi-image prompt and asks it to rank them
by similarity. The MLLM ranking is then soft-blended with the original IRRA
similarity scores.

Usage:
    python visual_rerank/infer.py \\
        --checkpoint logs/.../best.pth \\
        --config     logs/.../configs.yaml \\
        --mllm_dir   Qwen/Qwen2-VL-2B-Instruct \\
        --load_cache data/gallery_cache.pt \\
        --top_k      5 \\
        --alpha      0.5
"""

import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_parent_dir, "icl_rerank"))  # for: from mllm import MLLMs
sys.path.insert(0, _parent_dir)                              # for: from model import ...

import argparse
import json
import logging
import os.path as op
import re

import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.bases import ImageDataset, tokenize
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from model import build_model
from mllm import MLLMs
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.)
           for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
    )
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
                t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
                t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]


def print_rs(sims_dict, qids, pids, logger):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
    for key in sims_dict.keys():
        sims = sims_dict[key]
        rs = get_metrics(sims, qids, pids, f'{key}-t2i', False)
        table.add_row(rs)
    table.custom_format["R1"]   = lambda f, v: f"{v:.2f}"
    table.custom_format["R5"]   = lambda f, v: f"{v:.2f}"
    table.custom_format["R10"]  = lambda f, v: f"{v:.2f}"
    table.custom_format["mAP"]  = lambda f, v: f"{v:.2f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"
    logger.info('\n' + str(table))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_pretrained_model(cfg, checkpoint_path):
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = raw.get('model', raw)
    num_classes = next(
        (v.shape[0] for k, v in sd.items() if k.endswith('classifier.weight')), 11003
    )
    del raw, sd
    model = build_model(cfg, num_classes=num_classes)
    Checkpointer(model).load(f=checkpoint_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Caption encoding
# ---------------------------------------------------------------------------

class TextPureDataset(Dataset):
    def __init__(self, captions, text_length: int = 77, truncate: bool = True):
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return tokenize(
            self.captions[index],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )


def get_cap_embeds(model, captions, text_length, batch_size, device):
    cap_dataset = TextPureDataset(captions, text_length=text_length)
    loader = DataLoader(cap_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    model.eval()
    with torch.no_grad():
        for tokens in tqdm(loader, desc="Encoding captions", leave=False):
            tokens = tokens.to(device)
            feat = model.encode_text(tokens)
            feats.append(feat.cpu())
    return F.normalize(torch.cat(feats, 0), p=2, dim=1)


# ---------------------------------------------------------------------------
# Prompt + response parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batched MLLM reranking (multiple images per query in a single prompt)
# ---------------------------------------------------------------------------

def mllm_batch_rerank(llm_obj, sims, img_paths, captions, top_k, alpha, batch_size, logger):
    """Rerank top-k candidates for every query using MLLM; return updated sims tensor."""
    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info

    n_queries = len(captions)
    sims_reranked = sims.clone()
    topk_vals, topk_idx = torch.topk(sims, k=top_k, dim=1)  # [Q, k]

    n_batches = (n_queries + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="MLLM reranking"):
        start = b * batch_size
        end = min(start + batch_size, n_queries)
        b_idx  = topk_idx[start:end]   # [bs, k]
        b_vals = topk_vals[start:end]   # [bs, k]

        # Build one message per query: k images then the text prompt
        messages = []
        for i in range(end - start):
            content = []
            for j in range(top_k):
                content.append({
                    "type": "image",
                    "image": img_paths[b_idx[i][j].item()],
                    "min_pixels": 50176,
                    "max_pixels": 50176,
                })
            content.append({"type": "text", "text": _build_prompt(captions[start + i], top_k)})
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": content},
            ])

        prompts = [
            llm_obj.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_data = [process_vision_info(msg)[0] for msg in messages]
        inputs = [
            {"prompt": p, "multi_modal_data": {"image": img_d}}
            for p, img_d in zip(prompts, image_data)
        ]
        sampling_params = SamplingParams(temperature=0.01, max_tokens=128, skip_special_tokens=True)
        outputs = llm_obj.llm.generate(inputs, sampling_params=sampling_params)
        responses = [o.outputs[0].text for o in outputs]

        if b == 0:
            logger.info(f"Sample MLLM response: {responses[0]}")

        for i, response in enumerate(responses):
            q_idx       = start + i
            k_indices   = b_idx[i].numpy()
            irra_scores = b_vals[i].numpy().astype(float)

            mllm_order = _parse_ranking(response, top_k)  # 1-based, best first
            mllm_scores = np.zeros(top_k)
            for new_rank, orig_pos in enumerate(mllm_order):
                mllm_scores[orig_pos - 1] = (top_k - new_rank) / max(top_k - 1, 1)

            lo, hi = irra_scores.min(), irra_scores.max()
            irra_norm = (irra_scores - lo) / (hi - lo) if hi > lo else np.ones(top_k)

            combined = alpha * irra_norm + (1.0 - alpha) * mllm_scores
            for j in range(top_k):
                sims_reranked[q_idx, k_indices[j]] = combined[j]

    return sims_reranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visual reranking on CUHK-PEDES")
    parser.add_argument("--checkpoint",      required=True,  help="Path to best.pth")
    parser.add_argument("--config",          required=True,  help="Path to configs.yaml")
    parser.add_argument("--data_dir",        default="data")
    parser.add_argument("--mllm_dir",        default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--load_cache",      default=None,   help="Gallery feature cache .pt to load")
    parser.add_argument("--save_cache",      default=None,   help="Where to write gallery feature cache")
    parser.add_argument("--top_k",           type=int,   default=5,
                        help="Number of top IRRA candidates to rerank per query")
    parser.add_argument("--alpha",           type=float, default=0.5,
                        help="Blend weight: alpha*irra_norm + (1-alpha)*mllm_score")
    parser.add_argument("--tensor_parallel", type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--split",           default="test", choices=["test", "val"])
    parser.add_argument("--output_dir",      default="visual_rerank/output")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--num_queries",     type=int,   default=0,
                        help="Limit to first N queries for smoke testing (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(op.join(args.output_dir, "visual_rerank.log")),
        ],
    )
    logger = logging.getLogger("visual_rerank")

    # ---- device ----
    if args.device != "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "cpu" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # ---- IRRA model ----
    cfg = load_train_configs(args.config)
    cfg.training = False
    model = get_pretrained_model(cfg, args.checkpoint)
    model.to(device)
    img_size    = tuple(cfg.img_size) if hasattr(cfg, 'img_size') else (384, 128)
    text_length = getattr(cfg, 'text_length', 77)
    logger.info(f"IRRA model loaded on {device} | img_size={img_size} text_length={text_length}")

    # ---- dataset ----
    dataset    = CUHKPEDES(root=args.data_dir)
    split_data = dataset.test if args.split == 'test' else dataset.val
    captions      = split_data['captions']
    caption_pids  = split_data['caption_pids']
    ds_image_pids = split_data['image_pids']
    ds_img_paths  = split_data['img_paths']
    logger.info(f"Split '{args.split}': {len(captions)} captions, {len(ds_image_pids)} gallery images")

    # ---- gallery features ----
    if args.load_cache and op.exists(args.load_cache):
        logger.info(f"Loading gallery cache from {args.load_cache}")
        cache      = torch.load(args.load_cache, map_location='cpu', weights_only=False)
        gfeats     = cache['feats']
        raw_pids   = cache['pids']
        image_pids = raw_pids.tolist() if torch.is_tensor(raw_pids) else list(raw_pids)
        img_paths  = cache['paths']
    else:
        logger.info("Computing gallery features...")
        transform   = build_transforms(img_size)
        img_dataset = ImageDataset(ds_image_pids, ds_img_paths, transform)
        img_loader  = DataLoader(img_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        gfeats_list = []
        model.eval()
        with torch.no_grad():
            for pids, imgs in tqdm(img_loader, desc="Encoding gallery"):
                gfeats_list.append(model.encode_image(imgs.to(device)).cpu())
        gfeats     = F.normalize(torch.cat(gfeats_list, 0), p=2, dim=1)
        image_pids = ds_image_pids
        img_paths  = ds_img_paths
        if args.save_cache:
            os.makedirs(op.dirname(op.abspath(args.save_cache)), exist_ok=True)
            torch.save({'feats': gfeats, 'pids': image_pids, 'paths': img_paths}, args.save_cache)
            logger.info(f"Gallery cache saved to {args.save_cache}")

    logger.info(f"Gallery: {gfeats.shape[0]} images, feat dim={gfeats.shape[1]}")

    # ---- encode query captions ----
    logger.info("Encoding query captions...")
    qfeats = get_cap_embeds(model, captions, text_length, args.batch_size, device)

    # ---- baseline similarity ----
    sims_base = qfeats @ gfeats.t()  # [Q, G]
    qids = torch.tensor(caption_pids)
    gids = torch.tensor(image_pids)

    # ---- optional query limit for smoke testing ----
    if args.num_queries > 0:
        logger.info(f"Limiting to first {args.num_queries} queries (smoke test)")
        sims_base = sims_base[:args.num_queries]
        qids      = qids[:args.num_queries]
        captions  = captions[:args.num_queries]

    logger.info("Baseline metrics:")
    print_rs({'baseline': sims_base}, qids, gids, logger)

    if args.top_k == 0:
        logger.info("--top_k 0: skipping MLLM, done.")
        return

    # ---- MLLM ----
    logger.info(f"Loading MLLM from {args.mllm_dir} (tensor_parallel={args.tensor_parallel})")
    llm = MLLMs(model_dir=args.mllm_dir, tensor_parallel_size=args.tensor_parallel)

    # ---- rerank ----
    logger.info(f"Reranking top-{args.top_k} candidates per query (alpha={args.alpha})...")
    sims_reranked = mllm_batch_rerank(
        llm, sims_base, img_paths, captions,
        top_k=args.top_k, alpha=args.alpha,
        batch_size=args.batch_size, logger=logger,
    )

    logger.info("Reranked metrics:")
    print_rs({'reranked': sims_reranked}, qids, gids, logger)

    # ---- save config ----
    config_path = op.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {config_path}")


if __name__ == '__main__':
    main()
