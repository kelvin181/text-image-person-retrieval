# Text-to-Image Person Retrieval

Given a plain-English description of a person, this system searches a gallery of 34,000+ images and returns the most visually similar matches — ranked by confidence.

**Example:** *"a woman wearing a red jacket and black trousers"* → top-5 matching images from the gallery.

This is built on [IRRA](https://arxiv.org/abs/2303.12501) (CVPR 2023), fine-tuned on the [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) person re-identification dataset.

## How it works

1. **Text → vector**: the query is encoded by a transformer into a 512-dimensional embedding
2. **Images → vectors**: each gallery image is encoded by a Vision Transformer (ViT-B/16) into the same 512-dimensional space
3. **Ranking**: cosine similarity between the query vector and every image vector; top-k images are returned

Both encoders are trained jointly so that images and text descriptions of the same person land close together in vector space.

## Performance (CUHK-PEDES test set, 3,074 images)

| Rank-1 | Rank-5 | Rank-10 | mAP |
|--------|--------|---------|-----|
| 72.9% | 89.6% | 93.8% | 66.1% |

Rank-1 means the correct person is the top result; Rank-5 means they appear somewhere in the top 5.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the pretrained checkpoint

Download `best.pth` and `configs.yaml` from [Google Drive](https://drive.google.com/file/d/1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW/view?usp=share_link) and place them at:

```
logs/CUHK-PEDES/pretrained/
├── best.pth
└── configs.yaml
```

### 3. Prepare the dataset

Download CUHK-PEDES images via HuggingFace (requires ~4 GB disk space):

```bash
python prepare_data.py --data_dir data/
```

If you already have the images elsewhere, symlink them to avoid re-downloading:

```bash
python prepare_data.py --data_dir data/ --images-source /path/to/existing/images
```

### 4. Encode the gallery (one-time, ~30 min on CPU)

```bash
python encode_gallery.py \
  --checkpoint logs/CUHK-PEDES/pretrained/best.pth \
  --config     logs/CUHK-PEDES/pretrained/configs.yaml \
  --gallery_dir data \
  --save_cache  data/gallery_cache.pt
```

This encodes all 34,052 gallery images and saves their vectors to a cache file. You only need to do this once.

### 5. Run a query

```bash
python retrieve.py \
  --query      "a woman in a red jacket and black trousers" \
  --checkpoint logs/CUHK-PEDES/pretrained/best.pth \
  --config     logs/CUHK-PEDES/pretrained/configs.yaml \
  --load_cache data/gallery_cache.pt \
  --top_k 5 \
  --output_dir results/
```

This prints a ranked results table and copies the top-5 images to `results/`.

### Run multiple queries at once

```bash
python batch_query.py \
  --checkpoint logs/CUHK-PEDES/pretrained/best.pth \
  --config     logs/CUHK-PEDES/pretrained/configs.yaml \
  --cache      data/gallery_cache.pt \
  --n_queries  100
```

### Evaluate all test captions (CMC & mAP)

Encodes all 34,820 test captions and reports CMC@1/5/10, mAP, and mINP against the full gallery (~3 minutes on Apple Silicon):

```bash
python eval_all.py \
  --checkpoint logs/CUHK-PEDES/pretrained/best.pth \
  --config     logs/CUHK-PEDES/pretrained/configs.yaml \
  --cache      data/gallery_cache.pt
```

Results (34,820 queries, 34,052-image gallery):

| CMC@1 | CMC@5 | CMC@10 | mAP | mINP |
|-------|-------|--------|-----|------|
| 74.86% | 92.79% | 96.61% | 59.11% | 36.57% |

## Credits

> Jiang, D., Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval*. CVPR 2023. [arXiv:2303.12501](https://arxiv.org/abs/2303.12501)
