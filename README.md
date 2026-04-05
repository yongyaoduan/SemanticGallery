# SemanticGallery

![SemanticGallery demo](docs/assets/readme/semanticgallery-demo.gif)

SemanticGallery is a local-first semantic image search app for Apple Silicon. It builds a local search index for a folder of images and serves a lightweight web UI for search, preview, metadata inspection, and deletion.

- Local-first: the gallery stays on disk
- Semantic search for photos and screenshots
- First run prepares the model, local adaptation, and search index automatically
- Web UI supports preview, metadata inspection, and permanent delete
- Apple Silicon runtime built on MLX

## Requirements

- Apple Silicon
- [`uv`](https://github.com/astral-sh/uv)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`, `.heif`

## Quick Start

```bash
GALLERY_PATH=/absolute/path/to/gallery ./scripts/quickstart.sh
```

On the first run, SemanticGallery downloads the published public checkpoint from [Lucas20250626/semanticgallery-mlx-siglip2-stage1](https://huggingface.co/Lucas20250626/semanticgallery-mlx-siglip2-stage1), downloads the compact public anchor used by local adaptation from [Lucas20250626/semanticgallery-stage2-public-anchor](https://huggingface.co/datasets/Lucas20250626/semanticgallery-stage2-public-anchor), prepares a capped local adaptation set from the target gallery, runs the local adaptation step, builds the search index, and starts the web app.

Later runs reuse local artifacts unless the local adaptation input changes. On an Apple M4 MacBook Air, the demo album first run takes about one minute. Larger galleries scale with the number of images to encode.

Once startup finishes, open `http://127.0.0.1:36168` and start searching.

- Default URL: `http://127.0.0.1:36168`
- Startup log: `logs/runtime/semanticgallery_36168.log`
- PID file: `logs/runtime/semanticgallery_36168.pid`
- Stop the service: `kill "$(cat logs/runtime/semanticgallery_36168.pid)"`

## What Gets Written

- `logs/runtime/`: startup log and PID file for the running web service
- `logs/semanticgallery_private_data_adapted/`: local adaptation weights and training summary
- `deployment/search_config_gallery_mlx.json`: runtime search configuration
- `deployment/*_mlx_siglip2_embeddings.npy`, `deployment/*_mlx_siglip2.paths.txt`, `deployment/*_mlx_siglip2_skipped.json`: generated search index files for the selected gallery

## Delete Behavior

Deleting an image from the web UI permanently removes the file from the target gallery and refreshes the local search index. The app uses a temporary staging directory while it rewrites the index, but it is not a recycle bin or restore feature.

## Advanced Topics

- [Architecture](docs/architecture.md)
- [Data and Training](docs/training.md)
- [Benchmarks](docs/benchmarks.md)
- [Data Preparation](docs/data-preparation.md)
