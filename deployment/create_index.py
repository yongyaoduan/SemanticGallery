from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Write the pure-MLX search config for SemanticGallery.")
    parser.add_argument("--gallery-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--embeddings-file", default=None)
    parser.add_argument("--indexed-paths-file", default=None)
    parser.add_argument("--skipped-file", default=None)
    parser.add_argument("--file-state-file", default=None)
    parser.add_argument("--metadata-manifest", default=None)
    parser.add_argument("--config-output", default="./deployment/search_config.json")
    return parser.parse_args()


def resolve_artifacts(gallery_path: Path, embeddings_file: str | None, indexed_paths_file: str | None, skipped_file: str | None):
    deployment_dir = Path(__file__).resolve().parent
    stem = gallery_path.name
    embeddings_path = Path(embeddings_file).expanduser().resolve() if embeddings_file else deployment_dir / f"{stem}_mlx_siglip2_embeddings.npy"
    paths_path = Path(indexed_paths_file).expanduser().resolve() if indexed_paths_file else deployment_dir / f"{stem}_mlx_siglip2.paths.txt"
    skipped_path = Path(skipped_file).expanduser().resolve() if skipped_file else deployment_dir / f"{stem}_mlx_siglip2_skipped.json"
    return embeddings_path, paths_path, skipped_path


def main():
    args = parse_args()
    gallery_path = Path(args.gallery_path).expanduser().resolve()
    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery path not found: {gallery_path}")

    embeddings_path, paths_path, skipped_path = resolve_artifacts(
        gallery_path,
        args.embeddings_file,
        args.indexed_paths_file,
        args.skipped_file,
    )
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not paths_path.exists():
        raise FileNotFoundError(f"Indexed paths file not found: {paths_path}")

    payload = {
        "backend": "mlx_siglip2",
        "model_name": Path(args.model_path).expanduser().resolve().as_posix(),
        "weights_file": Path(args.weights_file).expanduser().resolve().as_posix() if args.weights_file else None,
        "model_precision": args.precision,
        "gallery_folder": gallery_path.name,
        "gallery_path": gallery_path.as_posix(),
        "indexed_paths_file": paths_path.as_posix(),
        "embeddings_file": embeddings_path.as_posix(),
        "skipped_images_file": skipped_path.as_posix(),
        "text_max_length": args.text_max_length,
    }
    if args.metadata_manifest:
        payload["metadata_manifest"] = Path(args.metadata_manifest).expanduser().resolve().as_posix()
    if args.file_state_file:
        payload["file_state_file"] = Path(args.file_state_file).expanduser().resolve().as_posix()

    config_output = Path(args.config_output).expanduser().resolve()
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"search_backend={payload['backend']}")
    print(f"gallery_path={payload['gallery_path']}")
    print(f"config_output={config_output}")
    print(f"embeddings_file={payload['embeddings_file']}")
    print(f"indexed_paths_file={payload['indexed_paths_file']}")
    if payload.get("weights_file"):
        print(f"weights_file={payload['weights_file']}")
    if payload.get("metadata_manifest"):
        print(f"metadata_manifest={payload['metadata_manifest']}")


if __name__ == "__main__":
    main()
