# SemanticGallery

![SemanticGallery demo](docs/assets/readme/semanticgallery-demo.gif)

SemanticGallery is a local-first semantic image search app for Apple Silicon. You pick a folder of images, the app indexes that folder locally, and you search it from a browser or the packaged macOS desktop app.

- Private images stay on disk. The gallery itself is not uploaded.
- Semantic search works across photos and screenshots.
- On the first run, SemanticGallery prepares the local runtime and downloads missing model assets with visible progress.
- The desktop app keeps Stage 2 manual. You choose a folder first, then run Stage 2 from Settings only when you want a folder-specific adaptation.
- The web UI supports text search, image search, similar-image search, preview, metadata inspection, batch selection, and permanent delete.
- The runtime is built on MLX for Apple Silicon.

## Requirements

- Apple Silicon
- [`uv`](https://github.com/astral-sh/uv)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`, `.heif`

## Desktop App

The first GitHub Release publishes a macOS desktop build.

- Install the latest `SemanticGallery-macos-arm64.dmg` from GitHub Releases. The release also keeps `SemanticGallery-macos-arm64.app.zip` as a fallback artifact.
- On the first launch, the desktop app creates `~/Library/Application Support/com.semanticgallery.desktop/`, prepares a local Python runtime, and downloads the MLX base model, the published Stage 1 checkpoint, and the small public Stage 2 anchor only when those assets are missing.
- Open **Settings** and click **Choose Folder** to pick a local album from Finder. The active folder search view is folder-scoped, but the shared SQLite index reuses embeddings by image content hash and stores a separate path row for every observed file.
- **Run Stage 2** appears in **Settings**. It starts only when you click it, and it refuses to run when the active folder has fewer than `100` supported images.
- The refresh icon forces an immediate rescan. The desktop app also runs a lightweight folder check every `5` seconds and skips the expensive path when nothing changed.

Maintainers can produce the signed desktop release locally with:

```bash
python scripts/build_desktop_release.py --require-developer-id --require-notarization
```

That command expects a `Developer ID Application` signing identity plus notarization credentials in the environment. It writes the final `.dmg` and `.app.zip` into `dist/release/`.

## Quick Start

```bash
GALLERY_DIR=/absolute/path/to/gallery ./scripts/quickstart.sh
```

On the first run, SemanticGallery does the following:

- creates `.venv/` and installs Python dependencies
- downloads the MLX SigLIP2 base model cache
- downloads the published retrieval checkpoint from [Lucas20250626/semanticgallery-mlx-siglip2-stage1](https://huggingface.co/Lucas20250626/semanticgallery-mlx-siglip2-stage1)
- downloads the small public reference set used to keep local adaptation stable from [Lucas20250626/semanticgallery-stage2-public-anchor](https://huggingface.co/datasets/Lucas20250626/semanticgallery-stage2-public-anchor)
- scans the target gallery and selects up to `100` local images for adaptation
- runs a short gallery-specific training step
- builds the gallery index and starts the web app

Private images do not leave the machine. The app goes online only for four things: Python packages, the MLX base model, the published retrieval checkpoint, and the small public reference set that local adaptation uses.

When startup succeeds, you should see:

- Default URL: `http://127.0.0.1:36168`
- Ready marker: `SemanticGallery is ready at http://127.0.0.1:36168`
- Startup log: `logs/runtime/semanticgallery_36168.log`
- PID file: `logs/runtime/semanticgallery_36168.pid`
- Browser view: one search box and a grid of results for the target gallery
- Stop the service: `kill "$(cat logs/runtime/semanticgallery_36168.pid)"`

If startup fails, `quickstart.sh` exits with a non-zero status and leaves the full log in `logs/runtime/`.

On later runs, the app rebuilds the full local manifest first and then applies three reuse checks:

- The capped local adaptation set stays fixed unless at least `10%` of its tracked files are missing.
- Stage 2 reruns only when one of these inputs changes: the published Stage 1 checkpoint, the capped local adaptation set, the tracked local image contents, or the Stage 2 hyperparameters.
- The gallery index synchronizes only when the gallery contents or the final weights change. Unchanged images stay in place. New or changed images are encoded when their path, file size, or `mtime_ns` changed. Deleted images are removed from the local index.

Deletes from the web UI update the local index immediately. Use `FORCE=1` only when you want to force a manual gallery re-encode:

```bash
FORCE=1 GALLERY_DIR=/absolute/path/to/gallery ./scripts/quickstart.sh
```

## What Gets Written

- `~/Library/Application Support/com.semanticgallery.desktop/index.sqlite3`: the desktop index store. It keeps `image_assets`, `image_embeddings`, `image_paths`, and `folder_states` so the app can reuse embeddings by content hash while still tracking every folder path separately.
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/`: the desktop runtime workspace copied from the bundled app resources
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.venv/`: local Python environment created for the desktop sidecar
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.cache/mlx/`: MLX SigLIP2 base model cache used by the desktop app
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.cache/semanticgallery/stage1/`: downloaded published Stage 1 checkpoint used by the desktop app
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.cache/semanticgallery/stage2_public_anchor/`: downloaded Stage 2 public reference set used by the desktop app
- `.venv/`: local Python environment created by `uv`
- `.cache/mlx/`: MLX SigLIP2 base model cache
- `.cache/semanticgallery/stage1/`: downloaded published Stage 1 checkpoint
- `.cache/semanticgallery/stage2_public_anchor/`: downloaded Stage 2 public reference set
- `datasets/private_gallery_local/<gallery-key>/full_manifest.jsonl`: full local manifest with absolute paths and weak labels
- `datasets/private_gallery_local/<gallery-key>/private_adapt_data.jsonl`: capped local adaptation subset
- `datasets/private_gallery_local/<gallery-key>/private_adapt_data_state.json`: tracked local adaptation rows, state for missing counts, and the content signature that Stage 2 reuse checks
- `logs/runtime/`: startup log and PID file for the running web service
- `logs/semanticgallery_private_data_adapted/<gallery-key>/`: local adaptation weights, a training history, a training summary, and `quickstart_state.json`
- `deployment/search_configs/<gallery-key>.json`: runtime search configuration
- `deployment/<gallery-key>_mlx_siglip2_embeddings.npy`, `deployment/<gallery-key>_mlx_siglip2.paths.txt`, `deployment/<gallery-key>_mlx_siglip2_skipped.json`, `deployment/<gallery-key>_mlx_siglip2_file_state.json`, `deployment/<gallery-key>_mlx_siglip2_bank_state.json`: generated search index and cache-state files for the selected gallery
- `deployment/.thumb_cache/`: cached JPEG thumbnails for the web UI
- `deployment/.delete_staging/`: temporary files used while delete rewrites the local index

`<gallery-key>` is a stable key derived from the absolute gallery path. It keeps one gallery from overwriting another gallery's manifests, adapted weights, caches, or search configs.

## Delete Behavior

Deleting an image from the web UI permanently removes the file from the target gallery and refreshes the local search index. The app uses a temporary staging directory while it rewrites the index, but nothing there acts as a recycle bin or a restore path.

## Advanced Topics

- [Architecture](docs/architecture.md)
- [Data Preparation](docs/data-preparation.md)
- [Training](docs/training.md)
- [Benchmarks](docs/benchmarks.md)
- [Runtime Reference](docs/reference.md)
- [Privacy and Limits](docs/privacy.md)
