"""
ICL THI (Test-time Human-centered Interaction) reranking on CUHK-PEDES.

Implements the multi-round iterative query refinement from the CVPR 2025 ICL paper:
  1. Anchor Location  – MLLM Yes/No: does this image match the text?
  2. Human-centered VQA – 15 fine-grained attribute questions about the anchor image
  3. Caption Aggregation – original query + VQA answers → refined caption
  4. Re-embed refined caption, blend with base similarity

Adapted from ICL/2025-CVPR-ICL/vllm_infer_ICL.py.

Usage:
    python icl_rerank/infer.py \\
        --checkpoint logs/.../best.pth \\
        --config     logs/.../configs.yaml \\
        --mllm_dir   Qwen/Qwen2-VL-2B-Instruct \\
        --load_cache data/gallery_cache.pt \\
        --rounds     5
"""

import sys
import os

# Make parent project importable from any working directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)   # for: from mllm import MLLMs
sys.path.insert(0, _parent_dir)   # for: from model import ..., from datasets import ...

import argparse
import base64
import copy
import json
import logging
import os.path as op
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
# Verbatim from ICL/2025-CVPR-ICL/vllm_infer_ICL.py lines 30–60
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


# ---------------------------------------------------------------------------
# Verbatim from ICL lines 62–68
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Verbatim from ICL lines 70–87
# ---------------------------------------------------------------------------
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        import requests
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Verbatim from ICL lines 195–217
# ---------------------------------------------------------------------------
def process_cap_(caps):
    tmps = []
    for c in caps:
        c = c.split('\n')
        tmp = []
        for cc in c:
            if ':' in cc:
                continue
            try:
                cc = cc.split('. ')[1]
                if 'Yes, ' in cc:
                    cc = cc.replace('Yes, ', '')
                if 'No, ' in cc:
                    cc = cc.replace('No, ', '')
                cc = cc[:1].upper() + cc[1:]
                if cc[-1:] != '.':
                    cc += '.'
            except:
                cc = ''
            tmp.append(cc)
        tmps.append(tmp)
    return tmps


# ---------------------------------------------------------------------------
# Adapted from ICL lines 219–232 — logger passed as parameter
# ---------------------------------------------------------------------------
def print_rs(sims_dict, qids, pids, logger):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
    for key in sims_dict.keys():
        sims = sims_dict[key]
        rs = get_metrics(sims, qids, pids, f'{key}-t2i', False)
        table.add_row(rs)

    table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"
    logger.info('\n' + str(table))


# ---------------------------------------------------------------------------
# Adapted from ICL lines 234–258 — batch_size passed as parameter
# ---------------------------------------------------------------------------
def batch_infer(llm, b_prompts, images, batch_size, t=0.2):
    n_samples = len(b_prompts)
    n_batches = (n_samples - 1) // batch_size + 1
    results = []
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = n_samples if i == n_batches - 1 else (i + 1) * batch_size
        rs = llm.generate_response_multi_images(
            questions=b_prompts[start:end], images=images[start:end], temperature=t
        )
        if rs:
            print(rs[0])
        results += rs
    return results


def batch_infer_txt(llm, b_prompts, batch_size, t=0.2):
    n_samples = len(b_prompts)
    n_batches = (n_samples - 1) // batch_size + 1
    results = []
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = n_samples if i == n_batches - 1 else (i + 1) * batch_size
        rs = llm.generate_response_text_only(questions=b_prompts[start:end], temperature=t)
        if rs:
            print(rs[0])
        results += rs
    return results


# ---------------------------------------------------------------------------
# TextPureDataset — copied from ICL/2025-CVPR-ICL/datasets/bases.py lines 161–176
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
        caption = tokenize(
            self.captions[index],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return caption


# ---------------------------------------------------------------------------
# Model loading — adapted from ICL lines 143–153 per plan spec
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
# Gallery + query embedding — adapted from ICL lines 361–379 (non-RDE path)
# ---------------------------------------------------------------------------
def get_embeddings(model, img_paths, image_pids, captions, caption_pids,
                   img_size, batch_size, text_length=77, device='cuda'):
    """Encode gallery images and query captions; return (qfeats, gfeats) on CPU."""
    transform = build_transforms(img_size)
    img_dataset = ImageDataset(image_pids, img_paths, transform)
    img_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    gfeats = []
    model.eval()
    with torch.no_grad():
        for pids, imgs in tqdm(img_loader, desc="Encoding gallery"):
            imgs = imgs.to(device)
            feat = model.encode_image(imgs)
            gfeats.append(feat.cpu())
    gfeats = F.normalize(torch.cat(gfeats, 0), p=2, dim=1)

    qfeats = get_cap_embeds(model, captions, text_length, batch_size, device)
    return qfeats, gfeats


# ---------------------------------------------------------------------------
# Caption re-embedding — adapted from ICL lines 411–426 (non-RDE path)
# ---------------------------------------------------------------------------
def get_cap_embeds(model, captions, text_length, batch_size, device):
    """Encode captions via IRRA text encoder; return L2-normalized CPU tensor."""
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
# THI round — adapted from ICL lines 260–358
# All ICL globals become keys in `state` dict, mutated in-place.
# ---------------------------------------------------------------------------
def round_llm(state, round_idx, llm, batch_size, logger):
    k = state['n_rounds']
    sims_base = state['sims_base']
    gt_indexs = state['gt_indexs']
    ggt_indexs = state['ggt_indexs']
    rcaptions = state['rcaptions']
    rrcaptions = state['rrcaptions']
    decisions = state['decisions']
    captions = state['captions']
    img_paths = state['img_paths']
    xi = state['xi']

    n_samples = sims_base.size(0)
    n_batches = (n_samples - 1) // batch_size + 1

    agrmaxs = []
    for i in tqdm(range(n_batches), desc=f"Top-k lookup round {round_idx}"):
        start = i * batch_size
        end = n_samples if i == n_batches - 1 else (i + 1) * batch_size
        agrmaxs += sims_base[start:end].topk(dim=1, k=k)[1].numpy().tolist()

    ref_images = [[img_paths[j] for j in row] for row in agrmaxs]
    ref_indexs = [[j for j in row] for row in agrmaxs]

    prompt1 = """Can this text accurately describe the image?

Text: {cap}

Answer "Yes" or "No"."""

    prompt2 = """According to the pedestrian image, answer the following questions one by one:

1. The person is male or female?
2. What hairstyle does the person have, such as hair length and color?
3. What is this person wearing on his upper body? If clearly visible, what are the color, type, and sleeve length?
4. What are the characteristics of this person's pants? If clearly visible, what are the color, type, and trouser leg length?
5. Does this person have any patterns on his/her clothes or pants?
6. What are the characteristics of this person's shoes? If clearly visible, what are the color and style?
7. Does this person wear glasses? If clearly visible, what are the color and style?
8. Is this person wearing a scarf? If clearly visible, what are the color and style?
9. Does this person have something in his/her hand? If so, what is it and what color is it?
10. Does this person carry a backpack? If clearly visible, what are the color and style?
11. Does this person wear a hat? If clearly visible, what are the color and style?
12. Is this person wearing a belt or waistband?
13. What is this person doing?
14. What is the background?
15. Are there other people in the background of this person?"""

    prompt3 = """Aggregate the following subtexts into continuous and concise text sentences.

Example:
Subtexts: ['The person is wearing a black jacket with a white stripe on the sleeve.', 'The person is male.', 'The person has short brown hair.', 'The person is wearing a sleeveless striped shirt and a green tank top.', 'The person is wearing green pants.', 'The person is wearing green pants.', 'The person is wearing a red scarf around their neck.', 'The person is wearing a red hat on their head.', 'The background is an outdoor area with some structures and other people.']
Output: The man has short brown hair and is wearing a black jacket with a white stripe on the sleeve, a sleeveless striped shirt, a green tank top, green pants, a red scarf around his neck, and a red hat. The background features an outdoor area with some structures and other people.

Now let's get started.
Subtexts: {cap}

Output aggregated sentences without any explanation."""

    sims = [sims_base[i][ref_indexs[i][0]] for i, v in enumerate(gt_indexs)]
    conditions = [1 if rrcaptions[i] == captions[i] else 0 for i, v in enumerate(gt_indexs)]

    # Stage 1 — Anchor Location
    images_stage1, prompts_stage1 = [], []
    for i, v in enumerate(gt_indexs):
        if conditions[i] == 1:
            images_stage1.append(ref_images[i][round_idx])
            prompts_stage1.append(prompt1.format(cap=captions[i]))

    rss = batch_infer(llm, prompts_stage1, images_stage1, batch_size, t=0.01)
    rpl_ids = [i for i, v in enumerate(gt_indexs) if conditions[i] == 1]

    for j, ids in enumerate(rpl_ids):
        if 'yes' in rss[j].lower():
            decisions[round_idx][ids] = 1
        else:
            decisions[round_idx][ids] = 0

    rrpl_ids = []
    for j, ids in enumerate(rpl_ids):
        flg = 0
        for l in range(round_idx):
            if decisions[l][ids] == 1:
                flg += 1
        if decisions[round_idx][ids] == 1 and round_idx == 0 and sims[ids] > xi:
            gt_indexs[ids] = ref_indexs[ids][round_idx]
            rrpl_ids.append(ids)
        if flg == 0 and decisions[round_idx][ids] == 1 and round_idx > 0 and sims[ids] <= xi:
            gt_indexs[ids] = ref_indexs[ids][round_idx]
            ggt_indexs[ids] = ref_indexs[ids][round_idx]
            rrpl_ids.append(ids)

    logger.info(f"Round {round_idx}: {len(rrpl_ids)} queries accepted for VQA")

    # Stage 2 — Human-centered VQA
    prompts_stage2 = [prompt2.format(cap=captions[v]) for v in rrpl_ids]
    images_stage2 = [img_paths[gt_indexs[v]] for v in rrpl_ids]
    rs = batch_infer(llm, prompts_stage2, images_stage2, batch_size, t=0.01)

    rs = process_cap_(rs)
    for i, v in enumerate(rs):
        rcaptions[rrpl_ids[i]] = v

    # Stage 3 — Caption Aggregation
    prompts_stage3 = [prompt3.format(cap=[captions[v]] + rcaptions[v]) for v in rrpl_ids]
    rs = batch_infer_txt(llm, prompts_stage3, batch_size, t=0.01)
    for i, v in enumerate(rs):
        rrcaptions[rrpl_ids[i]] = v


# ---------------------------------------------------------------------------
# Eval round — adapted from ICL lines 428–455
# ---------------------------------------------------------------------------
def eval_round(state, model, qids, pids, lam, text_length, batch_size, logger, device):
    rqfeats = get_cap_embeds(model, state['rrcaptions'], text_length, batch_size, device)
    sims_ = rqfeats @ state['gfeats'].t()

    for i, g in enumerate(state['gt_indexs']):
        if g > -1:
            tmp = sims_[i].clone()
            tmp[g] = 1.0
            sims_[i] = tmp
        else:
            sims_[i] = state['sims_base'][i].clone()

    sims_now = state['sims_base'] * lam + (1 - lam) * sims_

    sims_dict = {
        'sims_base': state['sims_base'],
        'sims_last': state['global_sims'],
        'sims_now': sims_now,
    }
    print_rs(sims_dict, qids, pids, logger)
    return sims_now


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ICL THI reranking on CUHK-PEDES")
    parser.add_argument("--checkpoint",      required=True,  help="Path to best.pth")
    parser.add_argument("--config",          required=True,  help="Path to configs.yaml")
    parser.add_argument("--data_dir",        default="data")
    parser.add_argument("--mllm_dir",        default="Qwen/Qwen2-VL-2B-Instruct",
                        help="HuggingFace model ID or local path for Qwen2-VL")
    parser.add_argument("--load_cache",      default=None,   help="Gallery cache .pt file to load")
    parser.add_argument("--save_cache",      default=None,   help="Where to write gallery cache")
    parser.add_argument("--rounds",          type=int, default=5,
                        help="Number of THI rounds (= top-k candidates examined)")
    parser.add_argument("--lam",             type=float, default=0.8,
                        help="λ blending weight: sims_base*λ + sims_refined*(1-λ)")
    parser.add_argument("--xi",              type=float, default=0.5,
                        help="Similarity threshold for round-0 anchor acceptance")
    parser.add_argument("--tensor_parallel", type=int, default=1,
                        help="vLLM tensor parallel size (number of GPUs)")
    parser.add_argument("--batch_size",      type=int, default=128)
    parser.add_argument("--split",           default="test", choices=["test", "val"])
    parser.add_argument("--output_dir",      default="icl_rerank/output")
    parser.add_argument("--device",          default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(op.join(args.output_dir, "icl.log")),
        ],
    )
    logger = logging.getLogger("ICL")

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
    img_size = tuple(cfg.img_size) if hasattr(cfg, 'img_size') else (384, 128)
    text_length = getattr(cfg, 'text_length', 77)
    logger.info(f"IRRA model loaded on {device} | img_size={img_size} text_length={text_length}")

    # ---- dataset ----
    dataset = CUHKPEDES(root=args.data_dir)
    split_data = dataset.test if args.split == 'test' else dataset.val
    captions = split_data['captions']
    caption_pids = split_data['caption_pids']
    ds_image_pids = split_data['image_pids']
    ds_img_paths = split_data['img_paths']
    logger.info(f"Split '{args.split}': {len(captions)} captions, {len(ds_image_pids)} gallery images")

    # ---- gallery features ----
    if args.load_cache and op.exists(args.load_cache):
        logger.info(f"Loading gallery cache from {args.load_cache}")
        cache = torch.load(args.load_cache, map_location='cpu')
        gfeats = cache['feats']
        raw_pids = cache['pids']
        image_pids = raw_pids.tolist() if torch.is_tensor(raw_pids) else list(raw_pids)
        img_paths = cache['paths']
    else:
        logger.info("Computing gallery features...")
        _, gfeats = get_embeddings(
            model, ds_img_paths, ds_image_pids, captions, caption_pids,
            img_size, args.batch_size, text_length=text_length, device=device,
        )
        image_pids = ds_image_pids
        img_paths = ds_img_paths
        if args.save_cache:
            os.makedirs(op.dirname(op.abspath(args.save_cache)), exist_ok=True)
            torch.save({'feats': gfeats, 'pids': image_pids, 'paths': img_paths}, args.save_cache)
            logger.info(f"Gallery cache saved to {args.save_cache}")

    logger.info(f"Gallery: {gfeats.shape[0]} images, feat dim={gfeats.shape[1]}")

    # ---- encode all query captions ----
    logger.info("Encoding query captions...")
    qfeats = get_cap_embeds(model, captions, text_length, args.batch_size, device)

    # ---- baseline similarity ----
    sims_base = qfeats @ gfeats.t()   # [Q, G]

    qids = torch.tensor(caption_pids)
    gids = torch.tensor(image_pids)

    logger.info("Baseline metrics:")
    print_rs({'baseline': sims_base}, qids, gids, logger)

    if args.rounds == 0:
        logger.info("--rounds 0: skipping LLM rounds, done.")
        return

    # ---- THI state ----
    N = len(captions)
    state = {
        'rrcaptions':  copy.deepcopy(captions),
        'rcaptions':   [[] for _ in captions],
        'gt_indexs':   [-1 for _ in captions],
        'ggt_indexs':  [-1 for _ in captions],
        'decisions':   np.zeros((args.rounds, N)),
        'sims_base':   sims_base,
        'global_sims': sims_base.clone(),
        'gfeats':      gfeats,
        'captions':    captions,
        'img_paths':   img_paths,
        'xi':          args.xi,
        'n_rounds':    args.rounds,
    }

    # ---- load MLLM ----
    logger.info(f"Loading MLLM from {args.mllm_dir} (tensor_parallel={args.tensor_parallel})")
    llm = MLLMs(model_dir=args.mllm_dir, tensor_parallel_size=args.tensor_parallel)

    # ---- THI loop ----
    for round_idx in range(args.rounds):
        logger.info("=" * 20 + f" Round {round_idx + 1}/{args.rounds} " + "=" * 20)
        round_llm(state, round_idx, llm, args.batch_size, logger)
        sims_now = eval_round(state, model, qids, gids, args.lam, text_length,
                              args.batch_size, logger, device)
        state['global_sims'] = sims_now

    # ---- save config ----
    config_path = op.join(args.output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    logger.info(f"Config saved to {config_path}")


if __name__ == '__main__':
    main()
