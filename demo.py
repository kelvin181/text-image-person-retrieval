"""
Visualize top-k retrieval results as a side-by-side image grid.

Usage:
    python demo.py \
        --query "a man in blue jeans and a white shirt" \
        --checkpoint logs/CUHK-PEDES/.../best.pth \
        --config    logs/CUHK-PEDES/.../configs.yaml \
        --gallery_dir data/CUHK-PEDES \
        --top_k 5 \
        --output results/demo.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from retrieve import retrieve


def visualize(results, query_text, output_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Query: "{query_text}"', fontsize=13, y=1.02, wrap=True)

    for ax, item in zip(axes, results):
        img = mpimg.imread(item['path'])
        ax.imshow(img)
        ax.set_title(f"Rank {item['rank']}\nPID {item['pid']}\n{item['score']:.3f}",
                     fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved visualization to: {output_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize text-to-image retrieval results")
    parser.add_argument("--query", required=True,
                        help="Natural language description of a person")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pth checkpoint file")
    parser.add_argument("--config", required=True,
                        help="Path to configs.yaml saved during training")
    parser.add_argument("--gallery_dir", default="data",
                        help="Root directory containing CUHK-PEDES/ (default: data)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results to show (default: 5)")
    parser.add_argument("--output", default="results/demo.png",
                        help="Output image path (default: results/demo.png)")
    parser.add_argument("--save_cache", default=None,
                        help="Path to save gallery feature cache (.pt)")
    parser.add_argument("--load_cache", default=None,
                        help="Path to load gallery feature cache (.pt)")
    parser.add_argument("--device", default="cuda",
                        help="Device to run inference on (default: cuda)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # reuse retrieve() — set output_dir=None to skip file copying
    args.output_dir = None
    results = retrieve(args)
    visualize(results, args.query, args.output)
