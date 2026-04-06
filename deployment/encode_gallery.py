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
    parser.add_argument("--file-state-output", default=None)
    return parser.parse_args()


def default_output_paths(
    gallery_path: Path,
    embeddings_output: str | None,
    paths_output: str | None,
    skipped_output: str | None,
    file_state_output: str | None,
):
    stem = gallery_path.name
    deployment_dir = Path(__file__).resolve().parent
    embeddings_path = Path(embeddings_output).expanduser().resolve() if embeddings_output else deployment_dir / f"{stem}_mlx_siglip2_embeddings.npy"
    paths_path = Path(paths_output).expanduser().resolve() if paths_output else deployment_dir / f"{stem}_mlx_siglip2.paths.txt"
    skipped_path = Path(skipped_output).expanduser().resolve() if skipped_output else deployment_dir / f"{stem}_mlx_siglip2_skipped.json"
    file_state_path = Path(file_state_output).expanduser().resolve() if file_state_output else deployment_dir / f"{stem}_mlx_siglip2_file_state.json"
    return embeddings_path, paths_path, skipped_path, file_state_path


def build_file_state_row(gallery_path: Path, image_path: Path) -> dict[str, int | str]:
    stat = image_path.stat()
    return {
        "relative_path": image_path.relative_to(gallery_path).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_previous_embeddings(gallery_path: Path, embeddings_path: Path, paths_path: Path, file_state_path: Path):
    if not embeddings_path.exists() or not paths_path.exists():
        return {}, {}, [], False

    indexed_paths = [line for line in paths_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = np.load(embeddings_path)
    payload = None
    bootstrapped_state = False
    if file_state_path.exists():
        payload = json.loads(file_state_path.read_text(encoding="utf-8"))
    else:
        payload = []
        bootstrapped_state = True
        for indexed_path in indexed_paths:
            path = Path(indexed_path).expanduser().resolve()
            try:
                path.relative_to(gallery_path)
            except ValueError:
                payload.append(None)
                continue
            if not path.exists():
                payload.append(None)
                continue
            payload.append(build_file_state_row(gallery_path, path))

    if not isinstance(payload, list):
        return {}, {}, [], False
    if matrix.shape[0] != len(indexed_paths) or len(indexed_paths) != len(payload):
        return {}, {}, [], False

    embedding_map = {}
    state_map = {}
    for indexed_path, state_row, embedding in zip(indexed_paths, payload, matrix):
        resolved_path = Path(indexed_path).expanduser().resolve().as_posix()
        embedding_map[resolved_path] = np.asarray(embedding, dtype=np.float32)
        if isinstance(state_row, dict):
            state_map[resolved_path] = state_row
    return embedding_map, state_map, indexed_paths, bootstrapped_state


def encode_paths(model, processor, image_paths: list[Path], batch_size: int):
    encoded = {}
    skipped = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding gallery"):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                images.append(open_rgb_image(path))
                valid_paths.append(path)
            except Exception as exc:
                skipped.append({"path": str(path), "error": str(exc)})

        if not images:
            continue

        inputs = processor(images=images, return_tensors="mlx")
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds = l2_normalize(image_embeds)
        mx.eval(image_embeds)
        batch_embeddings = np.asarray(image_embeds, dtype=np.float32)
        for path, embedding in zip(valid_paths, batch_embeddings):
            encoded[path.resolve().as_posix()] = embedding

    return encoded, skipped


def main():
    args = parse_args()
    gallery_path = Path(args.gallery_path).expanduser().resolve()
    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery path not found: {gallery_path}")

    embeddings_path, paths_path, skipped_path, file_state_path = default_output_paths(
        gallery_path=gallery_path,
        embeddings_output=args.embeddings_output,
        paths_output=args.paths_output,
        skipped_output=args.skipped_output,
        file_state_output=args.file_state_output,
    )
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    paths_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    file_state_path.parent.mkdir(parents=True, exist_ok=True)

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

    previous_embeddings, previous_state, previous_paths, bootstrapped_state = load_previous_embeddings(
        gallery_path=gallery_path,
        embeddings_path=embeddings_path,
        paths_path=paths_path,
        file_state_path=file_state_path,
    )
    previous_path_set = {Path(path).expanduser().resolve().as_posix() for path in previous_paths}

    current_state = {}
    encode_queue = []
    reused_paths = []
    for image_path in image_paths:
        resolved_path = image_path.resolve().as_posix()
        state_row = build_file_state_row(gallery_path, image_path)
        current_state[resolved_path] = state_row
        if previous_state.get(resolved_path) == state_row and resolved_path in previous_embeddings:
            reused_paths.append(resolved_path)
        else:
            encode_queue.append(image_path)

    removed_paths = sorted(previous_path_set - set(current_state))
    print(f"reused_images={len(reused_paths)}")
    print(f"changed_or_new_images={len(encode_queue)}")
    print(f"removed_images={len(removed_paths)}")
    if bootstrapped_state:
        print("bootstrapped_file_state=true")

    encoded_map, skipped_images = encode_paths(model, processor, encode_queue, args.batch_size)

    final_paths = []
    final_embeddings = []
    final_state = []
    for image_path in image_paths:
        resolved_path = image_path.resolve().as_posix()
        if resolved_path in previous_embeddings and previous_state.get(resolved_path) == current_state[resolved_path]:
            embedding = previous_embeddings[resolved_path]
        else:
            embedding = encoded_map.get(resolved_path)
        if embedding is None:
            continue
        final_paths.append(resolved_path)
        final_embeddings.append(np.asarray(embedding, dtype=np.float32))
        final_state.append(current_state[resolved_path])

    if not final_embeddings:
        raise RuntimeError("No image embeddings were generated.")

    matrix = np.stack(final_embeddings, axis=0).astype("float32")
    np.save(embeddings_path, matrix)
    paths_payload = "\n".join(str(path) for path in final_paths)
    if paths_payload:
        paths_payload += "\n"
    paths_path.write_text(paths_payload, encoding="utf-8")
    skipped_path.write_text(json.dumps(skipped_images, indent=2, ensure_ascii=False), encoding="utf-8")
    file_state_path.write_text(json.dumps(final_state, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"encoded_images={len(final_paths)}")
    print(f"skipped_images={len(skipped_images)}")
    print(f"embedding_dim={matrix.shape[1]}")
    print(f"embeddings_output={embeddings_path}")
    print(f"paths_output={paths_path}")
    print(f"skipped_output={skipped_path}")
    print(f"file_state_output={file_state_path}")


if __name__ == "__main__":
    main()
