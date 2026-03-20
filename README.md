# Text-to-Image Person Retrieval

Given a plain-English description of a person, this system searches a gallery of images and returns the most visually similar matches — ranked by confidence.

**Example:** *"a woman wearing a red jacket and black trousers"* → top-5 matching images from the gallery.

Built on [IRRA](https://arxiv.org/abs/2303.12501) (CVPR 2023), fine-tuned on [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description). Optionally re-ranks results using a multimodal LLM (Qwen2-VL).

## How it works

Both a text query and a set of gallery images are encoded into the same 512-dimensional vector space. Retrieval is cosine similarity — images whose vectors are closest to the query vector are returned first.

The model is based on CLIP (ViT-B/16) but improves on it in three ways for person re-ID:

- **Input resolution** — images are resized to 384×128 (tall, narrow person crops) instead of CLIP's default 224×224 square
- **SDM loss** — replaces CLIP's InfoNCE with Similarity Distribution Matching, which handles multiple images of the same person in a training batch without treating them as negatives
- **MLM + ID losses** — masked language modelling (predict masked words using the paired image) and an identity classifier head both push the embeddings to be more discriminative across person identities

## Performance on CUHK-PEDES test split

34,820 captions as queries, 3,074 gallery images, 1,510 identities.

| R@1 | R@5 | R@10 | mAP | mINP |
|-----|-----|------|-----|------|
| 73.4% | 89.8% | 93.7% | 66.1% | 50.2% |

## Dataset

CUHK-PEDES contains 40,206 images of 13,003 people, each annotated with multiple natural language descriptions written by crowd workers. The structure is:

```
Person (identity)
├── Image 1  →  4–28 captions describing this image
├── Image 2  →  4–28 captions describing this image
└── ...
```

At evaluation time, each caption is a query. A result is correct if the returned image shows the same person (same identity ID), regardless of which image or caption it was paired with.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Checkpoint

Place the pretrained checkpoint and config at:

```
logs/CUHK-PEDES/pretrained/
├── best_real.pth
└── configs_orig.yaml
```

### 3. Dataset

Place the CUHK-PEDES dataset at:

```
data/CUHK-PEDES/
├── imgs/
└── reid_raw.json
```

The BPE tokenizer vocab must stay at `data/bpe_simple_vocab_16e6.txt.gz`.

### 4. Encode the gallery (one-time)

Run a single query with `--save_cache` to encode all gallery images and save them to disk. This only needs to be done once (~30 min on CPU, ~2 min on GPU):

```bash
python retrieve.py \
  --query       "a woman in a red jacket" \
  --checkpoint  logs/CUHK-PEDES/pretrained/best_real.pth \
  --config      logs/CUHK-PEDES/pretrained/configs_orig.yaml \
  --gallery_dir data \
  --save_cache  data/gallery_cache.pt \
  --output_dir  results/
```

## Usage

### Single query

```bash
python retrieve.py \
  --query       "a woman in a red jacket and black trousers" \
  --checkpoint  logs/CUHK-PEDES/pretrained/best_real.pth \
  --config      logs/CUHK-PEDES/pretrained/configs_orig.yaml \
  --load_cache  data/gallery_cache.pt \
  --top_k       5 \
  --output_dir  results/
```

Prints a ranked results table and copies the top-k images to `results/`.

### Evaluate all test captions

Encodes all 34,820 test captions and reports R@1/5/10, mAP, and mINP:

```bash
python eval_all.py \
  --checkpoint logs/CUHK-PEDES/pretrained/best_real.pth \
  --config     logs/CUHK-PEDES/pretrained/configs_orig.yaml \
  --cache      data/gallery_cache.pt
```

Pass `--split val` or `--split all` to evaluate on other splits.

### Single query with VLM reranking

Retrieves top-k candidates with IRRA, then asks Qwen2-VL to rerank them:

```bash
python rerank.py \
  --query        "a woman in a red jacket" \
  --checkpoint   logs/CUHK-PEDES/pretrained/best_real.pth \
  --config       logs/CUHK-PEDES/pretrained/configs_orig.yaml \
  --rerank_model Qwen/Qwen2-VL-7B-Instruct \
  --load_cache   data/gallery_cache.pt \
  --top_k        10 \
  --output_dir   results/reranked/
```

### Evaluate reranking on a query sample

Compares IRRA baseline vs MLLM-reranked metrics on a random subset of queries:

```bash
python eval_rerank.py \
  --checkpoint   logs/CUHK-PEDES/pretrained/best_real.pth \
  --config       logs/CUHK-PEDES/pretrained/configs_orig.yaml \
  --cache        data/gallery_cache.pt \
  --rerank_model Qwen/Qwen2-VL-7B-Instruct \
  --num_queries  100 \
  --top_k        10
```

## Repository structure

```
eval_all.py       — batch evaluation: all test captions → R@k, mAP, mINP
eval_rerank.py    — compare IRRA baseline vs VLM-reranked on a query sample
rerank.py         — single-query retrieval + Qwen2-VL reranking
retrieve.py       — single-query retrieval (no reranking)
model/
  clip_model.py   — CLIP backbone (ViT-B/16, adapted for 384×128 input)
  build.py        — IRRA model: wraps CLIP, adds ID classifier + MLM heads
datasets/
  cuhkpedes.py    — parses reid_raw.json into train/val/test splits
  bases.py        — ImageDataset, tokenize()
  build.py        — build_transforms()
utils/
  simple_tokenizer.py  — BPE tokenizer (vocab at data/bpe_simple_vocab_16e6.txt.gz)
  checkpoint.py        — load/save model weights
  iotools.py           — config loading, image reading
```

## Credits

> Jiang, D., Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval*. CVPR 2023. [arXiv:2303.12501](https://arxiv.org/abs/2303.12501)
