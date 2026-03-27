# Text-to-Image Person Retrieval

IRRA-based text-to-image person retrieval on CUHK-PEDES.

## Key Entry Points

- `train.py` — Fine-tune the model. Run via `bash run_train.sh`
- `test.py` — Evaluate on CUHK-PEDES test set: `python test.py --config_file logs/.../configs.yaml`
- `retrieve.py` — Top-k retrieval for a text query (primary deliverable)
- `demo.py` — Save a matplotlib image grid of retrieval results

## Dataset

CUHK-PEDES must be placed at:
```
data/CUHK-PEDES/
├── imgs/
└── reid_raw.json
```

The BPE tokenizer vocab is at `data/bpe_simple_vocab_16e6.txt.gz` — do not move it (the tokenizer at `utils/simple_tokenizer.py` hardcodes this relative path).

## Model Architecture

- Backbone: CLIP ViT-B/16 (`model/clip_model.py`)
- Main model: `IRRA` class in `model/build.py` — exposes `encode_image()` and `encode_text()`
- Losses: SDM + MLM + ID (`model/objectives.py`)

## Retrieval Workflow

```python
# retrieve.py does this:
gallery_feats = F.normalize(model.encode_image(imgs), p=2, dim=1)  # [N, 512]
query_feat   = F.normalize(model.encode_text(tokens), p=2, dim=1)  # [1, 512]
similarity   = query_feat @ gallery_feats.T                         # [1, N]
top_k_idx    = torch.topk(similarity[0], k=5).indices
```

## Dependencies

Install with `pip install -r requirements.txt`. Requires CUDA for training; CPU works for inference but is slow.

## Checkpoints

Training saves checkpoints to `logs/CUHK-PEDES/<timestamp>_<name>/`. The best checkpoint is `best.pth`; training config is `configs.yaml` in the same directory.

## NeSI (Mahuika HPC)

**Remote path:** `/nesi/nobackup/uoa04685/jehc016/Projects/text-image-person-retrieval/`
**SSH host:** `mahuika`
**SLURM account:** `uoa04685`
**Job scripts:** `jobs/` (desc_rerank.sl, visual_rerank.sl, icl_rerank.sl)
**Job logs:** `jobs/logs/<jobname>_<jobid>.out` / `.err`

### Syncing local changes to NeSI

```bash
# From local project root:
rsync -avz desc_rerank/ mahuika:/nesi/nobackup/uoa04685/jehc016/Projects/text-image-person-retrieval/desc_rerank/
rsync -avz visual_rerank/ mahuika:/nesi/nobackup/uoa04685/jehc016/Projects/text-image-person-retrieval/visual_rerank/
rsync -avz icl_rerank/ mahuika:/nesi/nobackup/uoa04685/jehc016/Projects/text-image-person-retrieval/icl_rerank/
```

### Submitting and monitoring jobs

```bash
# Submit (from NeSI project dir):
ssh mahuika "cd /nesi/nobackup/uoa04685/jehc016/Projects/text-image-person-retrieval && sbatch jobs/desc_rerank.sl"

# Or submit and capture job ID:
JOB1=$(sbatch --parsable jobs/desc_rerank.sl)

# Monitor:
squeue --me

# View output (replace JOBID):
tail -50 jobs/logs/desc_rerank_JOBID.out
tail -50 jobs/logs/desc_rerank_JOBID.err
```

### Environment notes

- Python venv: `venv/` (activate with `source venv/bin/activate`)
- HF cache redirected to nobackup (home dir is small): `HF_HOME=/nesi/nobackup/uoa04685/jehc016/.cache/huggingface`
- vLLM cache: `VLLM_CACHE_ROOT=/nesi/nobackup/uoa04685/jehc016/.cache/vllm`
- Both are set in the .sl job scripts already
- Gallery feature cache: `data/gallery_cache.pt` (shared across all rerankers)
- Pretrained checkpoint: `logs/CUHK-PEDES/pretrained/best_real.pth`

### Rerankers

Three MLLM-based rerankers using Qwen2-VL via vLLM (extra deps: `vllm`, `qwen_vl_utils`, `transformers>=4.40.0`):
- `desc_rerank/` — generate image descriptions, rank by text similarity to query
- `icl_rerank/` — iterative binary VQA to refine query embedding
- `visual_rerank/` — send top-k images in single multi-image prompt

## Adapted from IRRA

This repo is a standalone adaptation of [IRRA](https://arxiv.org/abs/2303.12501). Changes from upstream:
- `datasets/build.py`: CUHK-PEDES only (removed ICFG-PEDES, RSTPReid)
- `train.py`: Fixed `WORLD_SIZE` env var handling for single-GPU runs
- `test.py`: Removed hardcoded config path default
- Added `retrieve.py` and `demo.py`
