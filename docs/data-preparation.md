# Data Preparation

SemanticGallery has two data-preparation paths:

- the default path used by `quickstart.sh` for gallery-specific adaptation
- the full retraining path used only when reproducing or replacing the published Stage 1 checkpoint

## Default Path

```bash
PRIVATE_GALLERY_DIR=/absolute/path/to/gallery ./scripts/prepare_data.sh
```

This command scans the user gallery and writes:

- `datasets/private_gallery_local/full_manifest.jsonl`
- `datasets/private_gallery_local/private_adapt_data.jsonl`

`full_manifest.jsonl` contains one row per usable image in the target gallery. `private_adapt_data.jsonl` is the capped subset used by the default Stage 2 adaptation path. The cap is `100` rows. If the gallery has fewer usable images, it keeps the available rows.

### Schema

Each JSONL row uses the same schema:

| Field | Meaning |
| --- | --- |
| `image_path` | Absolute path to the image on local disk |
| `captions` | Weak labels derived from path-based heuristics |
| `split` | Deterministic `train` or `val` assignment |
| `source` | Source label written into the manifest |

The default Stage 2 loss does not optimize on private captions. The private rows keep the shared JSONL schema so the same loaders can parse both public and local manifests, but the gallery-specific loss reads image paths and image augmentations, not private text labels. Those weak captions still matter at runtime because the search engine can use them for a small metadata-based ranking boost.

### Caption Rules

The local manifest builder does not run OCR or a caption model. It applies fixed rules:

- the first path segment under the gallery root
- alias expansion from the built-in folder map
- filename stem with `_` and `-` converted to spaces
- extra caption phrases for paths that look like screenshots
- extra caption phrases for paths that look like document or scan images

The built-in alias map covers these names:

- `Screenshots`
- `Camera`
- `Documents`
- `Favorites`
- `Downloads`

## Stage 2 Public Reference Set

The default path does not download the full public corpus again. It reuses a published `1000`-row public reference set:

- `500` Flickr30k rows
- `500` Screen2Words rows

The default Stage 2 path uses that small public reference set to keep a public text-image signal active without asking every user to download the full public corpus again.

## Full Retraining Preparation

```bash
PREPARE_PUBLIC_DATA=1 PRIVATE_GALLERY_DIR=/absolute/path/to/gallery ./scripts/prepare_data.sh
```

This path prepares the public training data for Stage 1 and the local files built from the user's gallery:

- `datasets/flickr30k/`
- `datasets/screen2words_train/`
- `datasets/screen2words_val/`
- `datasets/private_gallery_local/full_manifest.jsonl`
- `datasets/private_gallery_local/private_adapt_data.jsonl`

Use this path only when you want to reproduce or replace the published Stage 1 checkpoint.
