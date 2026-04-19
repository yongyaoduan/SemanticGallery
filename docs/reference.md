# Runtime Reference

This page collects the user-facing reference details for the current desktop app.

## Requirements

- Apple Silicon
- macOS 15.0 or later
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`, `.heif`

## Workspace

- Search input: accepts text or a pasted image
- Result limits: `25`, `50`, `100`, `250`
- Grid layout: five columns with system thumbnails
- Preview actions: similar-image search, metadata, delete, close
- Selection mode: supports manual selection, select all, and batch delete

If no folder is selected, the search field still accepts text and pasted images. The app simply returns no results and shows the empty-state prompt.

## Settings

- `Choose Folder`: selects the searchable library folder
- `Status`: shows `No folder selected`, `Indexing`, `Adapting`, or `Ready`
- Busy activity: shows percentage, elapsed time, and remaining time while indexing or adapting
- `Start Private Adaptation`: always stays clickable. If the current folder is not ready, the app shows a notice instead of disabling the button

## Search Scope And Identity

- Search results always come from the currently selected folder
- Embeddings are stored by content hash, not by file path
- Multiple folders can point at the same underlying image content without recomputing the shared embedding
- Deleting one visible result moves only that current file to Trash

## Local Paths

SemanticGallery writes its local state under `~/Library/Application Support/SemanticGallery/`:

- `library.sqlite`
- `install-state.json`
- `Artifacts/`
- `Models/`
- `Datasets/`
- `Caches/`
- `Logs/`

The app can read bundled assets from `SemanticGallery.app/Contents/Resources/SemanticGalleryArtifacts`, but user-specific state always stays in Application Support.

## Release Package

The desktop `dmg` already includes:

- the SigLIP2 base model files
- the published stage-1 checkpoint
- the public adaptation anchor

Normal use stays local after install. Search, index preparation, preview, delete, and private adaptation all run on your Mac. The only special first-launch step is the Finder right-click `Open` flow that macOS uses for unsigned apps.
