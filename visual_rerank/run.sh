#!/bin/bash
# Visual reranking — see visual_rerank/infer.py for full documentation.
#
# Additional pip deps (not in requirements.txt — vllm conflicts with some torch versions):
#   pip install "vllm>=0.4.0" qwen_vl_utils "transformers>=4.40.0"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

CHECKPOINT=logs/CUHK-PEDES/.../best.pth
CONFIG=logs/CUHK-PEDES/.../configs.yaml
MLLM_DIR=Qwen/Qwen2-VL-2B-Instruct  # downloaded automatically by vLLM on first run
CACHE=data/gallery_cache.pt

CUDA_VISIBLE_DEVICES=0 python visual_rerank/infer.py \
    --checkpoint      $CHECKPOINT \
    --config          $CONFIG \
    --mllm_dir        $MLLM_DIR \
    --load_cache      $CACHE \
    --top_k           5 \
    --alpha           0.5 \
    --batch_size      32 \
    --tensor_parallel 1 \
    --output_dir      visual_rerank/output

# Smoke test (first 10 queries only):
#   add --num_queries 10

# For dual-GPU: CUDA_VISIBLE_DEVICES=0,1 and --tensor_parallel 2
