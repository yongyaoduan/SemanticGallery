# SemanticGallery

![SemanticGallery demo](docs/assets/readme/semanticgallery-demo.gif)

SemanticGallery is a local-first semantic image search app for Apple Silicon. Point it at a folder of images, let it build a local index, and search that folder from a browser.

- Private images stay on disk. The gallery itself is not uploaded.
- Semantic search works across photos and screenshots.
- First run prepares the runtime, adapts to the target gallery, and builds the local index automatically.
- The web UI supports preview, metadata inspection, and permanent delete.
- The runtime is built on MLX for Apple Silicon.

## Requirements

- Apple Silicon
- [`uv`](https://github.com/astral-sh/uv)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`, `.heif`

## Quick Start

```bash
GALLERY_DIR=/absolute/path/to/gallery ./scripts/quickstart.sh
```

On the first run, SemanticGallery:

- creates `.venv/` and installs Python dependencies
- downloads the MLX SigLIP2 base model cache
- downloads the published retrieval checkpoint from [Lucas20250626/semanticgallery-mlx-siglip2-stage1](https://huggingface.co/Lucas20250626/semanticgallery-mlx-siglip2-stage1)
- downloads the small public reference set used to keep local adaptation stable from [Lucas20250626/semanticgallery-stage2-public-anchor](https://huggingface.co/datasets/Lucas20250626/semanticgallery-stage2-public-anchor)
- scans the target gallery, samples up to `100` local images, runs a short gallery-specific adaptation step, builds the gallery index, and starts the web app

Private images do not leave the machine. Network access is only used to download Python packages, the MLX base model, the published retrieval checkpoint, and the small public reference set used during local adaptation.

Success looks like this:

- Default URL: `http://127.0.0.1:36168`
- Ready marker: `SemanticGallery is ready at http://127.0.0.1:36168`
- Startup log: `logs/runtime/semanticgallery_36168.log`
- PID file: `logs/runtime/semanticgallery_36168.pid`
- Browser view: one search box and a grid of results for the target gallery
- Stop the service: `kill "$(cat logs/runtime/semanticgallery_36168.pid)"`

If startup fails, `quickstart.sh` exits with a non-zero status and leaves the full log in `logs/runtime/`.

Later runs reuse the existing gallery-specific adaptation and the existing gallery index. Deletes from the web UI update the local index immediately. If Stage 2 reruns with different local data or a different checkpoint, `quickstart.sh` now rebuilds the gallery index automatically. Use `FORCE=1` when gallery files change or when you want to rebuild the index manually:

```bash
FORCE=1 GALLERY_DIR=/absolute/path/to/gallery ./scripts/quickstart.sh
```

## What Gets Written

- `.venv/`: local Python environment created by `uv`
- `.cache/mlx/`: MLX SigLIP2 base model cache
- `.cache/semanticgallery/stage1/`: downloaded published Stage 1 checkpoint
- `.cache/semanticgallery/stage2_public_anchor/`: downloaded Stage 2 public reference set
- `datasets/private_gallery_local/full_manifest.jsonl`: full local manifest with absolute paths and weak labels
- `datasets/private_gallery_local/private_adapt_data.jsonl`: capped local adaptation subset
- `logs/runtime/`: startup log and PID file for the running web service
- `logs/semanticgallery_private_data_adapted/`: local adaptation weights, training summary, and `quickstart_state.json`
- `deployment/search_config_gallery_mlx.json`: runtime search configuration
- `deployment/*_mlx_siglip2_embeddings.npy`, `deployment/*_mlx_siglip2.paths.txt`, `deployment/*_mlx_siglip2_skipped.json`: generated search index files for the selected gallery
- `deployment/.thumb_cache/`: cached JPEG thumbnails for the web UI
- `deployment/.delete_staging/`: temporary files used while delete rewrites the local index

## Delete Behavior

Deleting an image from the web UI permanently removes the file from the target gallery and refreshes the local search index. The app uses a temporary staging directory while it rewrites the index, but it is not a recycle bin or restore feature.

## Advanced Topics

- [Architecture](docs/architecture.md)
- [Data Preparation](docs/data-preparation.md)
- [Training](docs/training.md)
- [Benchmarks](docs/benchmarks.md)
- [Runtime Reference](docs/reference.md)
- [Privacy and Limits](docs/privacy.md)
