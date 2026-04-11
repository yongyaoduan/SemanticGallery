# Privacy and Limits

## Privacy

- Private images stay on local disk.
- Gallery-specific adaptation runs on local files.
- By default the web app binds to `127.0.0.1`, so it is reachable only from the local machine.
- If you change `HOST`, the web app can become reachable from other machines on the same network.
- By default, network access is only used to download Python packages, the MLX SigLIP2 base model, the published Stage 1 checkpoint, and the Stage 2 public reference set.

## Local Files Written

- `~/Library/Application Support/com.semanticgallery.desktop/index.sqlite3` stores desktop `image_assets`, `image_embeddings`, `image_paths`, and `folder_states` rows. Embeddings are keyed by image content hash, while every observed path still keeps its own row.
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.cache/mlx/` stores the MLX SigLIP2 base model cache used by the desktop app
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.cache/semanticgallery/stage1/` stores the published Stage 1 checkpoint used by the desktop app
- `~/Library/Application Support/com.semanticgallery.desktop/runtime/.cache/semanticgallery/stage2_public_anchor/` stores the downloaded public Stage 2 reference set used by the desktop app
- `datasets/private_gallery_local/<gallery-key>/full_manifest.jsonl` stores absolute image paths and weak labels
- `datasets/private_gallery_local/<gallery-key>/private_adapt_data.jsonl` stores the capped local adaptation subset
- `datasets/private_gallery_local/<gallery-key>/private_adapt_data_state.json` stores the tracked local adaptation rows, counts of missing rows, and the content signature for Stage 2 reuse
- `logs/semanticgallery_private_data_adapted/<gallery-key>/weights.safetensors` stores the latest local Stage 2 weights
- `logs/semanticgallery_private_data_adapted/<gallery-key>/history.jsonl` stores per-epoch local adaptation history
- `logs/semanticgallery_private_data_adapted/<gallery-key>/summary.json` stores the latest local adaptation summary
- `logs/semanticgallery_private_data_adapted/<gallery-key>/quickstart_state.json` stores the gallery path, the Stage 1 checkpoint path, and the Stage 2 reuse signature used by quickstart
- `deployment/search_configs/<gallery-key>.json` stores the selected gallery path, model path, index paths, and metadata-manifest path
- `deployment/<gallery-key>_mlx_siglip2.paths.txt` stores absolute gallery paths for the current index
- `deployment/<gallery-key>_mlx_siglip2_embeddings.npy` stores the current gallery embedding bank
- `deployment/<gallery-key>_mlx_siglip2_skipped.json` stores skipped local files and error reasons
- `deployment/<gallery-key>_mlx_siglip2_file_state.json` stores per-image cache state for incremental gallery sync
- `deployment/<gallery-key>_mlx_siglip2_bank_state.json` stores gallery-level cache state for incremental gallery sync
- `deployment/.thumb_cache/` stores generated JPEG thumbnails for the web UI
- `deployment/.delete_staging/` stores temporary files while delete rewrites the local index

## Delete Behavior

- Delete from the web UI is permanent.
- The app uses a temporary staging directory only while it rewrites the index.
- There is no recycle bin or restore feature.

## Reindex Behavior

- In the desktop app, search stays scoped to the current folder even though embeddings are reused globally by content hash.
- In the desktop app, the refresh icon forces an immediate reconciliation pass and a lightweight background check runs every `5` seconds for the active folder.
- Deletes from the web UI update the local index immediately.
- If the Stage 2 adaptation weights change, `quickstart.sh` rebuilds the gallery index automatically before it starts the web app.
- If gallery files are added, removed, renamed, or modified between runs, `quickstart.sh` synchronizes the gallery index automatically on the next startup. Unchanged images are reused. Only files whose path, size, or `mtime_ns` changed are re-encoded. Deleted images are removed from the local index.
- Rerun `quickstart.sh` with `FORCE=1` only when you want to force a manual rebuild even though the current gallery and model state still match the cached index.

## Stage 2 Limit

- The default gallery-specific adaptation path keeps at most `100` local images.
- This cap keeps the local adaptation step short and limits overfitting to a small personal gallery.
- That capped set stays fixed until at least `10%` of its tracked files are missing from the current gallery.
