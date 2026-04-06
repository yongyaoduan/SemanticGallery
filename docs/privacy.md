# Privacy and Limits

## Privacy

- Private images stay on local disk.
- Gallery-specific adaptation runs on local files.
- By default the web app binds to `127.0.0.1`, so it is reachable only from the local machine.
- If you change `HOST`, the web app can become reachable from other machines on the same network.
- Default network access is used only to download Python packages, the MLX SigLIP2 base model, the published Stage 1 checkpoint, and the Stage 2 public reference set.

## Local Files Written

- `datasets/private_gallery_local/full_manifest.jsonl` stores absolute image paths and weak labels
- `datasets/private_gallery_local/private_adapt_data.jsonl` stores the capped local adaptation subset
- `logs/semanticgallery_private_data_adapted/quickstart_state.json` stores the gallery path, Stage 1 checkpoint path, and manifest hash used for quickstart reuse
- `deployment/search_config_gallery_mlx.json` stores the selected gallery path, model path, index paths, and metadata-manifest path
- `deployment/*_mlx_siglip2.paths.txt` stores absolute gallery paths for the current index
- `deployment/*_mlx_siglip2_skipped.json` stores skipped local files and error reasons
- `deployment/.thumb_cache/` stores generated JPEG thumbnails for the web UI
- `deployment/.delete_staging/` stores temporary files while delete rewrites the local index

## Delete Behavior

- Delete from the web UI is permanent.
- The app uses a temporary staging directory only while it rewrites the index.
- There is no recycle bin or restore feature.

## Reindex Behavior

- Deletes from the web UI update the local index immediately.
- If the Stage 2 adaptation weights change, `quickstart.sh` rebuilds the gallery index automatically before it starts the web app.
- If gallery files are added, removed, renamed, or modified between runs, `quickstart.sh` synchronizes the gallery index automatically on the next startup. Unchanged images are reused, new or changed images are encoded, and deleted images are removed from the local index.
- Rerun `quickstart.sh` with `FORCE=1` only when you want to force a manual rebuild even though the current gallery and model state still match the cached index.

## Stage 2 Limit

- The default gallery-specific adaptation path keeps at most `100` local images.
- This cap keeps the local adaptation step short and limits overfitting to a small personal gallery.
- That capped set stays fixed until at least `10%` of its tracked files are missing from the current gallery.
