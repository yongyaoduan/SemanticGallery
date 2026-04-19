# Data Preparation

SemanticGallery prepares data in two places: folder indexing and private adaptation.

## Folder Indexing

When you choose a library folder in `Settings`, the app:

1. Scans the selected folder recursively for supported image files.
2. Filters out unsupported paths.
3. Computes a stable content hash for each image.
4. Stores one embedding row per unique hash and one file-instance row per visible path.
5. Builds the folder-scoped search view used by the workspace.

This keeps duplicate image content from being encoded twice while still letting multiple folders point at the same underlying image.

## Private Adaptation Set

Private adaptation also starts from the selected folder in `Settings`.

- If the folder has fewer than `100` supported images, the app shows a notice and does not start training.
- If the folder is ready, the app keeps `100` private images from that folder.
- The app mixes those images with the bundled `1000`-image public anchor set.
- Each epoch keeps the same `100` private images and samples `100` public images without replacement from that `1000`-image pool.
- The shipped app runs `10` local epochs for one adaptation session.

## What Gets Stored

The desktop app writes its prepared data under `~/Library/Application Support/SemanticGallery/`:

- `library.sqlite`: content hashes, file instances, embeddings, encoder versions, and training runs
- `Datasets/`: local manifests and derived rows used by private adaptation
- `Models/Adapted/`: adapted encoder weights and summaries

The packaged app does not move or rename files inside your selected folder during preparation. It only reads those files, hashes them, and writes its own local state into Application Support.
