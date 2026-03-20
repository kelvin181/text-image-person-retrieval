# Text-to-Image Person Retrieval

Given a plain-English description of a person, this system searches a gallery of images and returns the most visually similar matches — ranked by confidence.

**Example:** *"a woman wearing a red jacket and black trousers"* → top-5 matching images from the gallery.

Built on [IRRA](https://arxiv.org/abs/2303.12501) (CVPR 2023), fine-tuned on [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description). Optionally re-ranks results using a multimodal LLM (Qwen2-VL).

## How it works

Both a text query and a set of gallery images are encoded into the same 512-dimensional vector space. Retrieval is cosine similarity — images whose vectors are closest to the query vector are returned first.

### Starting point: zero-shot CLIP

CLIP was trained on 400 million image-text pairs to align images and text in a shared embedding space. Applied zero-shot to CUHK-PEDES (no fine-tuning, just cosine similarity), it achieves roughly **44% Rank-1** — a reasonable starting point, but it wasn't built for person re-ID.

The core limitation is that CLIP was trained on diverse web images at 224×224, with one caption per image, and no concept of person identity. Person re-ID has different requirements: tall narrow crops, multiple images of the same person from different cameras, and descriptions that must distinguish fine-grained appearance details like clothing colour and accessories.

### What IRRA adds

IRRA fine-tunes the CLIP backbone on CUHK-PEDES with three targeted improvements:

- **Input resolution** — images are resized to 384×128 (tall, narrow person crops) instead of CLIP's 224×224 square. The ViT positional embeddings are interpolated to match. This preserves full-body detail across head, torso, and feet.

- **SDM loss** — CLIP's InfoNCE loss treats every non-matching pair in a batch as a negative. In a person re-ID batch, multiple images of the *same person* will appear — InfoNCE incorrectly penalises these. Similarity Distribution Matching replaces hard negatives with a soft label distribution derived from person IDs, so same-person pairs are pulled together proportionally rather than pushed apart.

- **MLM + ID losses** — two additional objectives force more discriminative embeddings. Masked Language Modelling masks words in a caption (e.g. *"a woman in a [MASK] jacket"*) and requires the model to predict them using the paired image via cross-attention — grounding words like colours and patterns in specific visual regions. The Identity loss adds a linear classifier predicting person ID from the embedding, directly supervising the model to separate different identities in vector space.

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
