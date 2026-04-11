# SemanticGallery macOS App Design

## Goal

Turn SemanticGallery into an installable macOS app that launches in its own desktop window, prepares its runtime on first launch with visible progress, lets the user choose a local album folder, builds and refreshes searchable embeddings incrementally, and keeps Stage 2 adaptation as a manual action in Settings.

## Product Direction

The shipped app is a desktop-first experience for Apple Silicon. The user installs a macOS app from GitHub Releases, opens it without touching a terminal, sees first-launch setup progress inside the app, picks one local album folder, and starts searching as soon as that folder is indexed.

Stage 2 adaptation is no longer part of startup. The app treats adaptation as an optional quality upgrade that the user starts manually from Settings. If the selected folder has fewer than 100 supported images, the app blocks adaptation and explains why.

The first release keeps online setup simple. The app bundle does not embed the MLX base model or the public Stage 2 anchor dataset. If those assets are missing on first launch, the app downloads them and shows step-level and overall progress in the UI.

## Chosen Architecture

### Desktop Shell

The app uses Tauri as the desktop shell.

- Tauri provides the macOS window, menus, native folder picker, app lifecycle hooks, background task orchestration, and packaged release artifacts.
- The front end stays HTML, CSS, and JavaScript so the existing search UI can be reshaped instead of replaced.
- The desktop shell owns startup state, Settings visibility, refresh controls, and progress presentation.

### Python Sidecar

The ML and search runtime stays in Python.

- The sidecar checks the runtime, downloads assets, scans galleries, hashes files, generates embeddings, encodes text and image queries, serves search APIs, and runs Stage 2 adaptation.
- Existing MLX-based model loading and search logic remain the core retrieval engine.
- Tauri talks to the sidecar through a local process boundary. The sidecar exposes a narrow command and event surface instead of exposing the current standalone browser-oriented startup script directly.

### Runtime Boundary

The current `quickstart.sh` flow is split into explicit app operations.

- `prepare_runtime`: verify Python availability, create or reuse the local environment, install dependencies if needed, download the base model if missing, and download the public Stage 2 anchor dataset if missing.
- `select_folder`: choose the active local album folder through the macOS folder picker.
- `sync_folder_index`: scan the selected folder, reconcile it against the local index store, and make the folder searchable without running Stage 2.
- `run_stage2_adaptation`: validate that the active folder has at least 100 supported images, run manual adaptation, then rebuild the active folder search view with the new encoder signature.

## User Experience

### First Launch

On first launch, the app opens a setup screen before the search UI.

The setup screen shows:

- a readable title for the current step
- a short status line that explains what the app is doing
- a progress bar for the current step
- a total progress bar across all setup steps
- a toggle that switches between readable status messages and raw logs

The setup flow is fixed to these steps:

1. Check local runtime
2. Prepare Python dependencies
3. Download or verify the base model
4. Download or verify the public Stage 2 anchor data
5. Finish setup

Download steps report byte-based progress whenever the source supports it. Install and extraction steps report subtask progress and still stream readable status messages even when an exact byte count is unavailable.

When setup completes, the app immediately opens the native macOS folder picker and requires the user to choose a local album folder before search becomes available.

### Main Window

The main window stays search-first.

- The search bar remains the primary control.
- A refresh icon button appears in the main toolbar. It triggers a manual folder reconciliation pass.
- A Settings button opens a Settings sheet or panel.
- The current folder name and path are visible in the chrome so the user always knows which local album is active.

### Settings

Settings contains the controls that change runtime behavior.

- `Choose Folder` opens the macOS folder picker and switches the active album.
- `Rebuild Index` forces a full reconciliation for the active folder.
- `Run Stage 2` starts manual adaptation for the active folder only.
- The Stage 2 section explains that adaptation is optional, local to the current folder, and requires at least 100 supported images.
- The runtime assets section shows whether the base model and public anchor dataset are ready.

### Stage 2 Validation

Before Stage 2 starts, the app counts supported images in the active folder.

- If the folder has at least 100 supported images, adaptation can start.
- If the folder has fewer than 100 supported images, the app blocks the action and shows a clear message in the UI.

The error message should follow this shape:

`This folder currently has N supported images. Stage 2 adaptation is only useful for folders with at least 100 images, so the app will skip adaptation for now.`

### Long-Running Tasks

Both folder indexing and Stage 2 adaptation surface progress in the UI.

- Each task shows the current phase, a progress indicator, and recent readable log lines.
- The user can continue reading the current results while a background refresh runs.
- If Stage 2 is running, the app disables starting another Stage 2 job for the same folder.

## Local Index Store

### Storage Choice

The app replaces the current folder-scoped `npy + text + json` artifact layout as the primary search store with a local SQLite database.

SQLite is the right fit for the first desktop release because it ships cleanly inside a local app, supports atomic updates, handles relational lookups between content hashes and file paths, and keeps the implementation simpler than a separate service.

### Hash Policy

The content key uses SHA-256 of the image file bytes.

SHA-256 is preferred over MD5 for this app because the index is meant to be a durable local source of truth, not just a fast collision-prone cache hint. Hashing cost is acceptable because hashing only happens for new or changed files.

### Schema

The index store has four core tables.

#### `image_assets`

One row per unique image content hash.

Fields:

- `content_hash`
- `byte_size`
- `created_at`
- `updated_at`
- optional image metadata such as width, height, and thumbnail cache state

#### `image_embeddings`

One row per `content_hash + encoder_signature`.

Fields:

- `content_hash`
- `encoder_signature`
- `embedding_blob`
- `embedding_dim`
- `created_at`

This table is the reusable vector store. If the same image content appears in multiple folders, the embedding is stored once per encoder signature and reused across paths.

#### `image_paths`

One row per observed file path.

Fields:

- `path_id`
- `absolute_path`
- `folder_path`
- `content_hash`
- `byte_size`
- `mtime_ns`
- `is_present`
- `last_scanned_at`

This table keeps path history independent from content identity. Two different paths can point to the same content hash.

#### `folder_states`

One row per folder the user has selected at least once.

Fields:

- `folder_path`
- `active_encoder_signature`
- `file_count`
- `total_bytes`
- `scan_signature`
- `last_scanned_at`
- `last_synced_at`

This table makes lightweight periodic change detection cheap.

## Search Model and Folder-Scoped Retrieval

### Encoder Signature

Every searchable embedding is tied to an encoder signature.

The encoder signature captures the model family, precision, and exact retrieval weights used to generate embeddings. The published Stage 1 weights and a locally adapted Stage 2 weights file therefore produce different encoder signatures.

### Active Search View

Search always targets the currently selected folder only.

At runtime, the sidecar builds an in-memory active search view for the current folder:

1. Load all `image_paths` rows for the active folder where `is_present = true`.
2. Join those rows to `image_embeddings` using each row's `content_hash` and the folder's `active_encoder_signature`.
3. Build a contiguous `float32` matrix and a row-to-path mapping for the active folder.
4. Reuse that matrix for text search, image search, and similar-image search until the folder view changes.

This design keeps embeddings globally reusable while making the search results folder-scoped.

### Retrieval Method

The first desktop release uses exact cosine similarity search over the active folder matrix.

This keeps the retrieval stack stable while the app grows new desktop behavior, incremental syncing, and manual adaptation support. Exact search also fits incremental updates better than introducing an approximate nearest-neighbor layer in the same release.

If future galleries grow large enough that exact search becomes a clear bottleneck, an ANN layer can be added later as a derived acceleration structure on top of the SQLite source of truth.

## Folder Sync and Index Refresh

### Lightweight Background Checks

The app runs a lightweight folder check every five seconds for the active folder.

The lightweight check does not touch the encoder. It only computes a cheap folder-level signature from the visible supported files, their counts, their aggregate bytes, and their path and `mtime` state. If the signature is unchanged, the app skips reconciliation.

### Manual Refresh

The main toolbar refresh icon forces an immediate reconciliation pass.

The user-facing meaning of refresh is simple: make the active folder index consistent with the actual folder contents. That includes:

- discovering new files
- noticing files deleted outside the app
- detecting files whose content changed in place
- filling in any missing embeddings for the active encoder
- removing stale path rows from the active search view

### Reconciliation Rules

When the app detects folder changes, it applies these rules:

- New path, known content hash, and an existing embedding for the active encoder: reuse that embedding and add or update the path row.
- New path, new content hash: hash the file, create or update the asset row, generate the embedding for the active encoder, then add the path row.
- Existing path, same size and `mtime_ns`: treat as unchanged and skip hashing.
- Existing path, changed size or `mtime_ns`: re-hash the file; if the content hash changed, update the path row to point to the new content identity.
- Missing path: mark the path row as not present and remove it from the active search view.

### Active View Updates

The app uses two update modes for the current folder matrix.

- Small changes: apply an incremental patch to the in-memory view.
- Large changes: rebuild the active folder matrix from SQLite.

The change threshold is implementation-controlled. The goal is predictable correctness first and lower rebuild cost second.

## Manual Stage 2 Adaptation

### Trigger

Stage 2 starts only when the user presses `Run Stage 2` in Settings.

No first-launch flow, app restart, folder switch, or background refresh should start Stage 2 automatically.

### Data Source

Stage 2 always uses the current active folder as its private local adaptation source.

The sidecar prepares or refreshes the capped private adaptation manifest for that folder, reuses the published public anchor dataset, and writes training outputs to the folder-specific adaptation area under the app data directory.

### Post-Training Behavior

When Stage 2 finishes successfully:

1. The sidecar creates a new encoder signature for the adapted weights.
2. The active folder switches its `active_encoder_signature` to that new value.
3. The app backfills embeddings for the active folder content hashes that do not yet have vectors under the new signature.
4. After the backfill completes, the app atomically swaps the active folder search view to the new matrix.

This prevents mixed results from old and new retrieval weights inside one folder.

## Packaging and Release

### Release Artifact

The first GitHub Release publishes a macOS installer artifact for Apple Silicon.

The preferred artifact is a signed `.dmg` when signing is available. If signing is not yet ready for the first public build, the fallback is a packaged `.app.zip` from the Tauri release pipeline.

### Bundled Contents

The installer includes:

- the Tauri desktop shell
- the front-end assets
- the Python sidecar entrypoint
- the code needed to create or reuse the local Python runtime

The installer does not include:

- the MLX base model weights
- the public Stage 2 anchor dataset

Those assets are downloaded on first launch if missing.

## Error Handling

The app should surface failures as readable desktop messages instead of shell-style failures.

- Runtime preparation failures stay on the setup screen and show the failed step, a readable explanation, and the recent logs.
- Folder selection failures leave the user on a safe idle state and let them retry.
- Index sync failures keep the last good active search view if one exists.
- Stage 2 failures keep the current folder on its previous encoder signature and do not replace the active search matrix.

## Testing Strategy

The implementation must ship with automated tests that cover the new behavior.

Required test areas:

- runtime progress event normalization for setup steps
- SQLite index-store operations for assets, embeddings, path rows, and folder states
- folder reconciliation for unchanged files, added files, removed files, and in-place edits
- embedding reuse across duplicate content in different folders
- folder-scoped search view construction
- Stage 2 validation for the fewer-than-100-images case
- post-Stage-2 encoder-signature switch behavior
- Tauri command integration for folder selection, refresh, and task state reporting

The final verification suite should include both Python-side tests and the Tauri-side test or build checks required by the desktop shell.
