"""
Description-Filter-Rerank pipeline on CUHK-PEDES test set.

For each query:
  1. IRRA retrieves top-N gallery candidates
  2. Qwen2-VL describes each candidate image ("Describe this person...")
  3. Qwen2-VL (text-only) filters out descriptions not relevant to the query
  4. Qwen2-VL (text-only) reranks the kept descriptions by relevance
  5. Kept images (LLM order) rise above filtered-out images and the rest of the
     gallery, which all retain their original IRRA ordering.

Usage:
    python desc_rerank/infer.py \\
        --checkpoint logs/.../best.pth \\
        --config     logs/.../configs.yaml \\
        --mllm_dir   Qwen/Qwen2-VL-2B-Instruct \\
        --load_cache data/gallery_cache.pt \\
        --top_k      20
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

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import DataLoader
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
# Ranking metrics — verbatim from icl_rerank/simple_infer.py
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


def get_metrics(similarity, qids, gids, n_, return_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
    )
    t2i_cmc = t2i_cmc.numpy()
    t2i_mAP = t2i_mAP.numpy()
    t2i_mINP = t2i_mINP.numpy()
    row = [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
           t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]
    return (row, indices) if return_indices else row


def print_rs(sims_dict, qids, pids, logger):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
    for key, sims in sims_dict.items():
        table.add_row(get_metrics(sims, qids, pids, f'{key}-t2i'))
    for col in ["R1", "R5", "R10", "mAP", "mINP", "rSum"]:
        table.custom_format[col] = lambda f, v: f"{v:.2f}"
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

class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, captions, text_length=77):
        self.captions = captions
        self.tokenizer = SimpleTokenizer()
        self.text_length = text_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return tokenize(self.captions[index], tokenizer=self.tokenizer,
                        text_length=self.text_length, truncate=True)


def encode_captions(model, captions, text_length, batch_size, device):
    ds = _TextDataset(captions, text_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    model.eval()
    with torch.no_grad():
        for tokens in tqdm(loader, desc="Encoding captions", leave=False):
            feats.append(model.encode_text(tokens.to(device)).cpu())
    return F.normalize(torch.cat(feats), p=2, dim=1)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_DESCRIBE_PROMPT = (
    "Describe this person's appearance in detail, including their clothing "
    "(colors, style, top and bottom garments), hair, build, and any accessories "
    "or distinguishing features."
)

_FILTER_TEMPLATE = (
    "Query: {query}\n"
    "Description: {desc}\n"
    "Is this description relevant to the person described in the query? "
    "Answer yes or no."
)

_RERANK_TEMPLATE = (
    "Query: {query}\n"
    "Below are descriptions of candidate images. "
    "Rank them from most to least relevant to the query.\n"
    "Descriptions:\n{desc_block}\n"
    "Output only the candidate numbers separated by commas (e.g. \"2, 1, 3\")."
)


def _parse_rerank(response, k):
    """Parse LLM rerank response into a 0-based index list (best first).

    Fills in any missing indices at the end to guarantee a complete ordering.
    """
    nums = re.findall(r'\b(\d+)\b', response)
    seen = set()
    order = []
    for n in nums:
        idx = int(n) - 1  # 1-based → 0-based
        if 0 <= idx < k and idx not in seen:
            seen.add(idx)
            order.append(idx)
    for i in range(k):
        if i not in seen:
            order.append(i)
    return order


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_desc_rerank(mllm, sims_base, img_paths, captions, top_k, batch_size, logger):
    """Describe → filter → rerank pipeline.

    Returns a new similarity tensor where kept images are scored above the rest.
    Filtered-out images and non-top-k gallery images retain their original IRRA
    ordering (their scores are not modified).
    """
    Q = sims_base.shape[0]
    _, topk_idx = torch.topk(sims_base, k=top_k, dim=1)  # [Q, top_k]

    # ------------------------------------------------------------------
    # Step 1: Describe all unique gallery images that appear in any top_k
    # ------------------------------------------------------------------
    unique_gal = sorted(set(topk_idx.flatten().tolist()))
    logger.info(f"Step 1: Describing {len(unique_gal)} unique gallery images "
                f"(top_k={top_k} across {Q} queries)...")

    unique_paths = [img_paths[i] for i in unique_gal]
    gal_to_desc = {}
    for start in tqdm(range(0, len(unique_paths), batch_size), desc="Describing images"):
        batch_paths = unique_paths[start:start + batch_size]
        batch_idxs  = unique_gal[start:start + batch_size]
        questions   = [_DESCRIBE_PROMPT] * len(batch_paths)
        responses   = mllm.generate_response_multi_images(questions, batch_paths)
        for gidx, desc in zip(batch_idxs, responses):
            gal_to_desc[gidx] = desc

    logger.info(f"  Sample description (gallery idx {unique_gal[0]}): "
                f"{gal_to_desc[unique_gal[0]][:120]}...")

    # ------------------------------------------------------------------
    # Step 2: Filter — for every (query, top_k candidate) pair, ask the
    # LLM whether the description is relevant to the query
    # ------------------------------------------------------------------
    logger.info(f"Step 2: Filtering {Q * top_k} (query, description) pairs...")

    filter_qs   = []
    filter_meta = []  # (q_idx, j_in_topk)
    for q in range(Q):
        for j in range(top_k):
            gidx = topk_idx[q, j].item()
            filter_qs.append(_FILTER_TEMPLATE.format(
                query=captions[q], desc=gal_to_desc[gidx]
            ))
            filter_meta.append((q, j))

    filter_resps = []
    for start in tqdm(range(0, len(filter_qs), batch_size), desc="Filtering"):
        filter_resps.extend(
            mllm.generate_response_text_only(filter_qs[start:start + batch_size])
        )

    # keep[q][j] = True when the j-th candidate in top_k was deemed relevant
    keep = [[False] * top_k for _ in range(Q)]
    for (q, j), resp in zip(filter_meta, filter_resps):
        keep[q][j] = resp.strip().lower().startswith("yes")

    n_kept = sum(sum(row) for row in keep)
    logger.info(f"  Kept {n_kept}/{Q * top_k} descriptions "
                f"({100.0 * n_kept / (Q * top_k):.1f}%)")
    logger.info(f"  Sample filter response: {filter_resps[0]!r}")

    # ------------------------------------------------------------------
    # Step 3: Rerank kept descriptions for queries with ≥2 surviving
    # ------------------------------------------------------------------
    logger.info("Step 3: Reranking kept descriptions...")

    # Separate queries needing a rerank call from those that don't
    needs_rerank   = []   # (q_idx, kept_positions_in_topk)
    no_rerank_kept = {}   # q_idx → kept_positions_in_topk (0 or 1 items)

    for q in range(Q):
        kept_pos = [j for j in range(top_k) if keep[q][j]]
        if len(kept_pos) >= 2:
            needs_rerank.append((q, kept_pos))
        else:
            no_rerank_kept[q] = kept_pos

    rerank_qs = []
    for q, kept_pos in needs_rerank:
        descs = [gal_to_desc[topk_idx[q, j].item()] for j in kept_pos]
        desc_block = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descs))
        rerank_qs.append(_RERANK_TEMPLATE.format(
            query=captions[q], desc_block=desc_block
        ))

    rerank_resps = []
    for start in tqdm(range(0, len(rerank_qs), batch_size), desc="Reranking"):
        rerank_resps.extend(
            mllm.generate_response_text_only(rerank_qs[start:start + batch_size])
        )

    if rerank_resps:
        logger.info(f"  Sample rerank response: {rerank_resps[0]!r}")

    # q_idx → 0-based order within kept_pos (best first)
    q_to_order = {}
    for (q, kept_pos), resp in zip(needs_rerank, rerank_resps):
        q_to_order[q] = _parse_rerank(resp, len(kept_pos))
    for q, kept_pos in no_rerank_kept.items():
        q_to_order[q] = list(range(len(kept_pos)))  # trivial order (0 or 1 item)

    # ------------------------------------------------------------------
    # Step 4: Merge — float kept images to the top by overwriting their
    # scores with values above the query's max IRRA score
    # ------------------------------------------------------------------
    sims_out = sims_base.clone()
    for q in range(Q):
        kept_pos = [j for j in range(top_k) if keep[q][j]]
        if not kept_pos:
            continue

        order = q_to_order[q]  # 0-based indices into kept_pos, best first
        max_score = sims_base[q].max().item()
        n_kept_q  = len(kept_pos)
        for rank_i, pos_in_kept in enumerate(order):
            j       = kept_pos[pos_in_kept]        # position within top_k
            gal_idx = topk_idx[q, j].item()         # gallery index
            # Assign decreasing scores above the query's max IRRA score
            sims_out[q, gal_idx] = max_score + (n_kept_q - rank_i)

    return sims_out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Description-Filter-Rerank on CUHK-PEDES"
    )
    parser.add_argument("--checkpoint",      required=True,  help="Path to best.pth")
    parser.add_argument("--config",          required=True,  help="Path to configs.yaml")
    parser.add_argument("--mllm_dir",        required=True,  help="Path to Qwen2-VL model dir")
    parser.add_argument("--data_dir",        default="data", help="Root containing CUHK-PEDES/")
    parser.add_argument("--load_cache",      default=None,   help="Gallery feature cache to load")
    parser.add_argument("--save_cache",      default=None,   help="Where to save gallery feature cache")
    parser.add_argument("--top_k",           type=int,   default=10,
                        help="Number of candidates to describe per query")
    parser.add_argument("--tensor_parallel", type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--split",           default="test", choices=["test", "val"])
    parser.add_argument("--output_dir",      default="desc_rerank/output")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--num_queries",     type=int,   default=0,
                        help="Limit to first N queries for smoke testing (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(op.join(args.output_dir, "desc_rerank.log")),
        ],
    )
    logger = logging.getLogger("desc_rerank")

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
    logger.info(f"IRRA loaded on {device} | img_size={img_size} text_length={text_length}")

    # ---- dataset ----
    dataset    = CUHKPEDES(root=args.data_dir)
    split_data = dataset.test if args.split == 'test' else dataset.val
    captions      = split_data['captions']
    caption_pids  = split_data['caption_pids']
    ds_image_pids = split_data['image_pids']
    ds_img_paths  = split_data['img_paths']
    logger.info(f"Split '{args.split}': {len(captions)} captions, "
                f"{len(ds_image_pids)} gallery images")

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
        gfeats     = F.normalize(torch.cat(gfeats_list), p=2, dim=1)
        image_pids = ds_image_pids
        img_paths  = ds_img_paths
        if args.save_cache:
            os.makedirs(op.dirname(op.abspath(args.save_cache)), exist_ok=True)
            torch.save({'feats': gfeats, 'pids': image_pids, 'paths': img_paths},
                       args.save_cache)
            logger.info(f"Gallery cache saved to {args.save_cache}")

    logger.info(f"Gallery: {gfeats.shape[0]} images, feat dim={gfeats.shape[1]}")

    # ---- encode query captions ----
    logger.info("Encoding query captions...")
    qfeats = encode_captions(model, captions, text_length, args.batch_size, device)

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
    logger.info(f"Loading MLLM from {args.mllm_dir} "
                f"(tensor_parallel={args.tensor_parallel})")
    mllm = MLLMs(model_dir=args.mllm_dir, tensor_parallel_size=args.tensor_parallel)

    # ---- describe → filter → rerank ----
    logger.info(f"Running describe-filter-rerank (top_k={args.top_k})...")
    sims_reranked = run_desc_rerank(
        mllm=mllm,
        sims_base=sims_base,
        img_paths=img_paths,
        captions=captions,
        top_k=args.top_k,
        batch_size=args.batch_size,
        logger=logger,
    )

    logger.info("Reranked metrics:")
    print_rs({'reranked': sims_reranked}, qids, gids, logger)

    # ---- save config ----
    config_out = op.join(args.output_dir, 'config.json')
    with open(config_out, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {config_out}")


if __name__ == '__main__':
    main()
