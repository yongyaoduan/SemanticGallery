# Reference

Set variables inline before the command:

```bash
NAME=value OTHER=value ./scripts/quickstart.sh
```

This page lists the user-facing environment variables exposed by the shell entrypoints in `scripts/`.

## Quick Start

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `GALLERY_PATH` | Absolute path to a local image folder. Required. | none | Selects the gallery to adapt, index, and serve. |
| `STAGE1_WEIGHTS` | Absolute path to a `weights.safetensors` file, or unset. | local Stage 1 if present, otherwise the published checkpoint | Overrides the starting checkpoint used by the Stage 2 adaptation step. |
| `HOST` | IP address or hostname string. Typical values: `127.0.0.1`, `0.0.0.0`. | `127.0.0.1` | Controls where the web app binds. `127.0.0.1` keeps it local. `0.0.0.0` exposes it to the local network. |
| `PORT` | Unused TCP port number. | `36168` | Changes the web URL and the runtime log and PID filenames. |
| `CONFIG_OUTPUT` | Path to a JSON file. | `deployment/search_config_gallery_mlx.json` | Changes where the generated search config is written. |
| `METADATA_MANIFEST` | Path to a local JSONL file, or unset. | `datasets/private_gallery_local/full_manifest.jsonl` | Supplies weak captions for metadata display and the small runtime metadata boost. |
| `MODEL_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | Controls precision for gallery encoding and query encoding. `bfloat16` is the normal fast path. `float32` is slower and more conservative. |
| `RUNTIME_DIR` | Directory path. | `logs/runtime` | Controls where `quickstart.sh` writes the startup log and PID file. |
| `LOG_FILE` | File path. | `logs/runtime/semanticgallery_<port>.log` | Overrides the startup log location. |
| `PID_FILE` | File path. | `logs/runtime/semanticgallery_<port>.pid` | Overrides the PID file location. |
| `STARTUP_TIMEOUT_SECONDS` | Positive integer. | `300` | Changes how long `quickstart.sh` waits for the ready marker before failing. |
| `FORCE` | `0` or `1`. | `0` | `1` reruns gallery encoding and rewrites the search config instead of reusing existing index files. |
| `ENCODE_BATCH_SIZE` | Positive integer. | `8` | Controls gallery-encoding batch size. Larger values can improve throughput but use more memory. |

## Direct Deployment

These variables apply when `deploy_best.sh` is run directly.

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `GALLERY_PATH` | Absolute path to a local image folder. Required. | none | Selects the gallery to index and serve. |
| `CONFIG_OUTPUT` | Path to a JSON file. | `deployment/search_config_gallery_mlx.json` | Changes where the generated search config is written. |
| `METADATA_MANIFEST` | Path to a local JSONL file, or unset. | `datasets/private_gallery_local/full_manifest.jsonl` | Supplies weak captions for metadata display and the small runtime metadata boost. |
| `HOST` | IP address or hostname string. Typical values: `127.0.0.1`, `0.0.0.0`. | `127.0.0.1` | Controls where the web app binds. |
| `PORT` | Unused TCP port number. | `36168` | Changes the web URL and the runtime log and PID filenames. |
| `MODEL_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | Controls precision for gallery encoding and query encoding. |
| `MODEL_WEIGHTS` | Absolute path to a `weights.safetensors` file, or unset. | latest local adaptation if present, otherwise Stage 1 | Overrides the weights used for gallery encoding and query encoding. |
| `FORCE` | `0` or `1`. | `0` | `1` reruns gallery encoding instead of reusing existing index files. |
| `ENCODE_BATCH_SIZE` | Positive integer. | `8` | Controls gallery-encoding batch size. Larger values can improve throughput but use more memory. |

## Data Preparation

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `PRIVATE_GALLERY_PATH` | Absolute path to a local image folder. | required unless manifests already exist | Builds `full_manifest.jsonl` and `private_adapt_data.jsonl` from the user's gallery. |
| `PREPARE_PUBLIC_DATA` | `0` or `1`. | `0` | `1` also downloads and prepares Flickr30k plus Screen2Words for full Stage 1 retraining. `0` keeps the default local-only preparation path. |
| `FORCE` | `0` or `1`. | `0` | Only affects the public Stage 1 datasets here. `1` refreshes them even if the local copies already meet the row threshold. |

## Gallery-Specific Adaptation

These variables apply to `adapt_best.sh`. In the shipped default Stage 2 path, the text tower stays frozen.

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `FINAL_RUN_NAME` | Directory name under `logs/`. | `semanticgallery_private_data_adapted` | Changes where Stage 2 writes weights, summary files, and history. |
| `STAGE1_WEIGHTS` | Absolute path to a `weights.safetensors` file, or unset. | local Stage 1 if present, otherwise the published checkpoint | Overrides the starting checkpoint for Stage 2. |
| `MAX_EPOCHS_STAGE2` | Positive integer. | `1` | Controls how many Stage 2 epochs are run. |
| `TRAIN_BATCH_SIZE` | Positive integer. | `4` | Controls the public text-image batch size used during Stage 2. |
| `TRAIN_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | Controls Stage 2 training precision. `bfloat16` is the normal fast path. `float32` is slower and more conservative. |
| `VISION_UNFREEZE_LAST_N` | Integer `>= 0`. | `2` | Controls how many vision blocks remain trainable. `0` freezes the vision tower. Larger values tune more of the encoder and use more memory. |
| `LR_STAGE2` | Positive float. | `5e-6` | Controls the Stage 2 learning rate. Larger values adapt faster but increase drift risk. |
| `PRIVATE_BATCH_SIZE` | Positive integer. | `8` | Controls how many local images are used per private Stage 2 step. |
| `PRIVATE_REPEATS_PER_EPOCH` | Positive integer. | `2` | Controls how many passes Stage 2 makes over the capped local set in one epoch. |
| `PRIVATE_INSTANCE_WEIGHT` | Non-negative float. | `0.3` | Weights the image-instance loss. Larger values pull harder toward gallery-specific invariance. |
| `PRIVATE_DISTILL_WEIGHT` | Non-negative float. | `0.15` | Weights the distillation loss. Larger values keep the adapted image encoder closer to the Stage 1 teacher. |
| `MAX_TRAIN_STEPS` | Positive integer, or unset. | unset | Caps training steps for smoke tests or short experiments. |
| `MAX_VAL_STEPS` | Positive integer, or unset. | unset | Caps validation steps for smoke tests or short experiments. |

## Full Retraining

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `PUBLIC_RUN_NAME` | Directory name under `logs/`. | `semanticgallery_public_stage1` | Changes where Stage 1 writes weights, summary files, and history. |
| `FINAL_RUN_NAME` | Directory name under `logs/`. | `semanticgallery_public_plus_private_data` | Changes where the final Stage 1 -> Stage 2 run writes weights, summary files, and history. |
| `MAX_EPOCHS_STAGE1` | Positive integer. | `1` | Controls how many Stage 1 epochs are run. |
| `MAX_EPOCHS_STAGE2` | Positive integer. | `1` | Controls how many Stage 2 epochs run after Stage 1. |
| `TRAIN_BATCH_SIZE` | Positive integer. | `4` | Controls the public text-image batch size in both stages. |
| `TRAIN_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | Controls training precision for both stages. |
| `TEXT_UNFREEZE_LAST_N` | Integer `>= 0`. | `2` | Controls how many text blocks remain trainable during Stage 1. `0` freezes the text tower. Larger values tune more of the text encoder and use more memory. |
| `VISION_UNFREEZE_LAST_N` | Integer `>= 0`. | `2` | Controls how many vision blocks remain trainable in both stages. |
| `LR_STAGE1` | Positive float. | `1e-5` | Controls the Stage 1 learning rate. |
| `LR_STAGE2` | Positive float. | `5e-6` | Controls the Stage 2 learning rate after Stage 1. |
| `PRIVATE_BATCH_SIZE` | Positive integer. | `8` | Controls how many local images are used per private Stage 2 step. |
| `PRIVATE_REPEATS_PER_EPOCH` | Positive integer. | `2` | Controls how many passes Stage 2 makes over the capped local set in one epoch. |
| `PRIVATE_INSTANCE_WEIGHT` | Non-negative float. | `0.3` | Weights the image-instance loss in Stage 2. |
| `PRIVATE_DISTILL_WEIGHT` | Non-negative float. | `0.15` | Weights the distillation loss in Stage 2. |
| `MAX_TRAIN_STEPS` | Positive integer, or unset. | unset | Caps training steps for smoke tests or short experiments. |
| `MAX_VAL_STEPS` | Positive integer, or unset. | unset | Caps validation steps for smoke tests or short experiments. |

## Common Operations

Rebuild the gallery index after adding or changing files, or after rerunning gallery-specific adaptation with different weights:

```bash
FORCE=1 GALLERY_PATH=/absolute/path/to/gallery ./scripts/quickstart.sh
```

Stop the running service:

```bash
kill "$(cat logs/runtime/semanticgallery_36168.pid)"
```
