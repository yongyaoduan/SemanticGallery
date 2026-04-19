# Privacy and Limits

## Privacy

- Private images stay on local disk.
- Indexing, search, preview, and private adaptation run on your Mac.
- The release package already contains the runtime assets it needs for normal use.
- After install, day-to-day search and adaptation do not require the app to fetch model files from the network.

## Local Files Written

SemanticGallery stores its own state under `~/Library/Application Support/SemanticGallery/`:

- `library.sqlite` stores content hashes, file instances, embeddings, encoder versions, and training runs
- `Artifacts/` stores the bundled model files copied into the user runtime
- `Models/Adapted/` stores the latest local adaptation weights and summaries
- `Datasets/` stores manifests and derived rows used by private adaptation
- `Caches/` stores generated thumbnail and runtime caches
- `Logs/` stores app logs

## Delete Behavior

- Delete from the workspace moves the current file to Trash.
- If the same image content exists in another selected folder path, those remaining paths stay intact.
- The app updates the database state and the in-memory search view immediately after delete.

## Reindex Behavior

- Selecting a folder scans only that folder for searchable results.
- If files are added, removed, renamed, or changed, the next index refresh updates the local database for that folder.
- When private adaptation finishes, the app rebuilds every stored embedding with the adapted encoder.

## Private Adaptation Limits

- The default gallery-specific adaptation path keeps `100` private images from the selected folder.
- The app mixes them with the bundled `1000`-image public anchor pool.
- If the current folder has fewer than `100` supported images, the app shows a notice and does not start training.
