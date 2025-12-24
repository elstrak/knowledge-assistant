import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ka.embeddings import EmbeddingConfig
from ka.indexing import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index from chunks.jsonl")
    parser.add_argument(
        "--chunks",
        default="dataset/processed/chunks.jsonl",
        help="Path to chunks.jsonl (default: dataset/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="dataset/index",
        help="Output directory for vector index (default: dataset/index)",
    )
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="SentenceTransformers model name (default: intfloat/multilingual-e5-small)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "hashing", "sentence-transformers"],
        help="Embeddings backend (default: auto). Use hashing to avoid torch/sentence-transformers.",
    )
    parser.add_argument(
        "--hashing-dim",
        type=int,
        default=4096,
        help="Dimensionality for hashing backend (default: 4096)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Debug: index only first N chunks",
    )
    args = parser.parse_args()

    chunks = os.path.abspath(os.path.expanduser(args.chunks))
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    cfg = EmbeddingConfig(model_name=args.model, backend=args.backend, hashing_dim=args.hashing_dim)

    print(f"[INFO] Building index from: {chunks}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Embedding model: {cfg.model_name}")
    print(f"[INFO] Embedding backend: {cfg.backend}")

    build_index(
        chunks_path=chunks,
        out_dir=out_dir,
        embed_cfg=cfg,
        max_chunks=args.max_chunks,
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()


