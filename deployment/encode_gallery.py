from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

import sys

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from mlx_pipeline import (
    DEFAULT_MLX_MODEL_PATH,
    collect_gallery_paths,
    l2_normalize,
    load_mlx_siglip_model,
    open_rgb_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode a local image gallery with the pure-MLX SigLIP2 stack.")
    parser.add_argument("--gallery-path", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MLX_MODEL_PATH.as_posix())
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embeddings-output", default=None)
    parser.add_argument("--paths-output", default=None)
    parser.add_argument("--skipped-output", default=None)
    return parser.parse_args()


def default_output_paths(gallery_path: Path, embeddings_output: str | None, paths_output: str | None, skipped_output: str | None):
    stem = gallery_path.name
    deployment_dir = Path(__file__).resolve().parent
    embeddings_path = Path(embeddings_output).expanduser().resolve() if embeddings_output else deployment_dir / f"{stem}_mlx_siglip2_embeddings.npy"
    paths_path = Path(paths_output).expanduser().resolve() if paths_output else deployment_dir / f"{stem}_mlx_siglip2.paths.txt"
    skipped_path = Path(skipped_output).expanduser().resolve() if skipped_output else deployment_dir / f"{stem}_mlx_siglip2_skipped.json"
    return embeddings_path, paths_path, skipped_path


def main():
    args = parse_args()
    gallery_path = Path(args.gallery_path).expanduser().resolve()
    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery path not found: {gallery_path}")

    embeddings_path, paths_path, skipped_path = default_output_paths(
        gallery_path=gallery_path,
        embeddings_output=args.embeddings_output,
        paths_output=args.paths_output,
        skipped_output=args.skipped_output,
    )
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    paths_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = collect_gallery_paths(gallery_path)
    if not image_paths:
        raise ValueError(f"No supported images found under {gallery_path}")
    print(f"gallery_path={gallery_path}")
    print(f"discovered_images={len(image_paths)}")
    print(f"model_path={Path(args.model_path).expanduser().resolve()}")
    print(f"precision={args.precision}")
    print("loading_model=true")

    model, processor = load_mlx_siglip_model(
        args.model_path,
        weights_file=args.weights_file,
        precision=args.precision,
        lazy=False,
    )
    print("loading_model=false")

    all_embeddings = []
    kept_paths = []
    skipped_images = []

    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Encoding gallery"):
        batch_paths = image_paths[start : start + args.batch_size]
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                images.append(open_rgb_image(path))
                valid_paths.append(path)
            except Exception as exc:
                skipped_images.append({"path": str(path), "error": str(exc)})

        if not images:
            continue

        inputs = processor(images=images, return_tensors="mlx")
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds = l2_normalize(image_embeds)
        mx.eval(image_embeds)
        all_embeddings.append(np.asarray(image_embeds, dtype=np.float32))
        kept_paths.extend(valid_paths)

    if not all_embeddings:
        raise RuntimeError("No image embeddings were generated.")

    matrix = np.concatenate(all_embeddings, axis=0).astype("float32")
    np.save(embeddings_path, matrix)
    paths_payload = "\n".join(str(path) for path in kept_paths)
    if paths_payload:
        paths_payload += "\n"
    paths_path.write_text(paths_payload, encoding="utf-8")
    skipped_path.write_text(json.dumps(skipped_images, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"encoded_images={len(kept_paths)}")
    print(f"skipped_images={len(skipped_images)}")
    print(f"embedding_dim={matrix.shape[1]}")
    print(f"embeddings_output={embeddings_path}")
    print(f"paths_output={paths_path}")
    print(f"skipped_output={skipped_path}")


if __name__ == "__main__":
    main()
