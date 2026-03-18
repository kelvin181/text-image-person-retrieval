# Text-to-Image Person Retrieval

Fine-tuned [IRRA](https://arxiv.org/abs/2303.12501) (Implicit Relation Reasoning and Aligning) on the CUHK-PEDES dataset for text-to-image person retrieval.

Given a natural language query such as *"a woman wearing a red jacket and black trousers"*, the system ranks gallery images by similarity and returns the top-5 matches.

## Architecture

- **Backbone:** CLIP ViT-B/16 (pre-trained on 400M image-text pairs)
- **Cross-modal module:** 4-layer multi-head cross-attention transformer
- **Training losses:** SDM (Similarity Distribution Matching) + MLM (Masked Language Modeling) + ID (identity classification)
- **Feature dimension:** 512
- **Image resolution:** 384 × 128 (person re-id standard)

## Expected Performance on CUHK-PEDES Test Set

| R@1 | R@5 | R@10 | mAP | mINP |
|-----|-----|------|-----|------|
| 73.38 | 89.93 | 93.71 | 66.13 | 50.24 |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CUHK-PEDES

Request access from the [official source](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) and place the files as follows:

```
data/
└── CUHK-PEDES/
    ├── imgs/
    │   ├── cam_a/
    │   └── cam_b/
    └── reid_raw.json
```

The BPE tokenizer vocabulary is already included at `data/bpe_simple_vocab_16e6.txt.gz`.

## Training

```bash
bash run_train.sh
```

This runs 60 epochs on a single GPU with the SDM+MLM+ID loss combination. Checkpoints and logs are saved to `logs/CUHK-PEDES/<timestamp>_irra/`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

Key hyperparameters (see `utils/options.py` for all defaults):

| Parameter | Value |
|-----------|-------|
| Backbone | ViT-B/16 |
| Batch size | 64 |
| Learning rate | 1e-5 (5× for cross-modal modules) |
| Epochs | 60 |
| Warmup epochs | 5 |
| Image size | 384 × 128 |
| Text length | 77 tokens |

## Evaluation

Run standard Rank-1/5/10, mAP, mINP evaluation on the CUHK-PEDES test set:

```bash
python test.py --config_file logs/CUHK-PEDES/<run_dir>/configs.yaml
```

## Retrieval

Return the top-5 gallery images for a text query:

```bash
python retrieve.py \
  --query "a woman in a red jacket and black jeans" \
  --checkpoint logs/CUHK-PEDES/<run_dir>/best.pth \
  --config    logs/CUHK-PEDES/<run_dir>/configs.yaml \
  --gallery_dir data \
  --top_k 5 \
  --output_dir results/
```

The top-5 images are saved to `results/` as `rank1_pid*.jpg` ... `rank5_pid*.jpg`, and a table is printed to the console.

**Gallery caching** — encoding ~3,074 gallery images takes ~10–30 seconds on GPU. Cache them for fast repeated queries:

```bash
# First run: encode and cache
python retrieve.py --query "..." --save_cache data/gallery_cache.pt ...

# Subsequent runs: load from cache (< 1 second)
python retrieve.py --query "..." --load_cache data/gallery_cache.pt ...
```

## Visualization

Save a side-by-side image grid of the top-5 results:

```bash
python demo.py \
  --query "a man in blue jeans and a white shirt" \
  --checkpoint logs/CUHK-PEDES/<run_dir>/best.pth \
  --config    logs/CUHK-PEDES/<run_dir>/configs.yaml \
  --gallery_dir data \
  --output results/demo.png
```

## Credits

Based on the IRRA model from:

> Jiang, D., Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval*. CVPR 2023.
> [arXiv:2303.12501](https://arxiv.org/abs/2303.12501)
