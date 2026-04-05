# Architecture

SemanticGallery uses a dual-encoder retrieval model and a split serving path: offline index build, online search, and delete sync.

## Model

![Model structure](assets/readme/model-structure.png)

The text tower encodes the query. The vision tower encodes each gallery image. Both outputs are L2-normalized into the same embedding space, so online retrieval is one text forward pass followed by vector scoring over the precomputed gallery index.

- Text input: query text tokenized by the SigLIP2 tokenizer
- Image input: RGB gallery image resized for the SigLIP2 vision tower
- Output: normalized embeddings in a shared vector space
- Why a dual encoder: the gallery can be encoded offline once, which keeps online latency low even for large folders

## Deployment

![Deployment architecture](assets/readme/deployment-architecture.png)

### Offline Build

The gallery encoder scans the target folder, filters supported image files, computes image embeddings in MLX, and writes the local search index. The runtime configuration records the gallery path, model path, index file paths, and optional metadata manifest.

### Online Search

The browser sends a text query to the local web app. The query encoder runs the SigLIP2 text tower in MLX and produces one query vector. The search engine scores that vector against the precomputed gallery embeddings and resolves the ranked paths into thumbnails, filenames, and capture-time metadata.

### Delete Sync

When the user deletes an image from the web UI, the app removes the file from the gallery, updates the file-backed index, and rewrites the metadata manifest if one is present. The next search sees the same state as the local folder.
