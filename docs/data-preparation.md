# Data Preparation

SemanticGallery uses different data paths for local adaptation and full retraining.

## Local Adaptation

```bash
PRIVATE_GALLERY_PATH=/absolute/path/to/gallery ./scripts/prepare_data.sh
```

This command scans the target gallery and writes:

- `datasets/private_gallery_local/full_manifest.jsonl`
- `datasets/private_gallery_local/private_adapt_data.jsonl`

`private_adapt_data.jsonl` is capped at `100` rows. If the gallery has fewer usable images, it keeps the available rows.

## Stage 2 Public Anchor

The default path does not download the full public corpus again. It reuses a fixed-seed public anchor published at [Lucas20250626/semanticgallery-stage2-public-anchor](https://huggingface.co/datasets/Lucas20250626/semanticgallery-stage2-public-anchor).

The public anchor contains:

- `500` Flickr30k rows
- `500` Screen2Words rows

Why this exists:

- avoid re-downloading the full public corpus during normal operation
- keep Stage 2 fast enough for local use
- preserve public text-image supervision while adapting to the target gallery

With the default `80/20` train / validation split, Stage 2 sees `800` public train rows and `200` public validation rows.

## Full Public Corpus

```bash
PREPARE_PUBLIC_DATA=1 PRIVATE_GALLERY_PATH=/absolute/path/to/gallery ./scripts/prepare_data.sh
```

This path prepares:

- Flickr30k under `datasets/flickr30k/`
- Screen2Words train under `datasets/screen2words_train/`
- Screen2Words val under `datasets/screen2words_val/`
- the same local adaptation manifests under `datasets/private_gallery_local/`

Use it only when you want to reproduce or replace the published Stage 1 checkpoint.
