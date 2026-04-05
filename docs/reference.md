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
| `STAGE1_WEIGHTS` | Absolute path to a `weights.safetensors` file, or unset. | local Stage 1 if present, otherwise the published checkpoint | If unset, `quickstart.sh` uses the local Stage 1 checkpoint when present and falls back to the published checkpoint otherwise. If set, it skips that resolution path and starts Stage 2 from the file you provided. |
| `HOST` | IP address or hostname string. Typical values: `127.0.0.1`, `0.0.0.0`. | `127.0.0.1` | `127.0.0.1` keeps the app reachable only from the local machine. `0.0.0.0` binds every interface and makes the app reachable from the local network. Any other address binds only that specific interface. |
| `PORT` | Unused TCP port number. | `36168` | Changes the web URL and the runtime log and PID filenames. |
| `CONFIG_OUTPUT` | Path to a JSON file. | `deployment/search_config_gallery_mlx.json` | Changes where the generated search config is written. |
| `METADATA_MANIFEST` | Path to a local JSONL file, or unset. | `datasets/private_gallery_local/full_manifest.jsonl` | If set, the runtime loads weak captions from that file for metadata display and the small metadata-based ranking boost. If unset, search still works, but the extra metadata boost and manifest-backed metadata display are disabled. |
| `MODEL_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | `bfloat16` is the normal fast path for gallery encoding and query encoding. `float32` uses more memory and runs slower, but is the more conservative option if you want full-precision inference. |
| `RUNTIME_DIR` | Directory path. | `logs/runtime` | Controls where `quickstart.sh` writes the startup log and PID file. |
| `LOG_FILE` | File path. | `logs/runtime/semanticgallery_<port>.log` | Overrides the startup log location. |
| `PID_FILE` | File path. | `logs/runtime/semanticgallery_<port>.pid` | Overrides the PID file location. |
| `STARTUP_TIMEOUT_SECONDS` | Positive integer. | `300` | A larger value gives the app more time to finish adaptation or startup before `quickstart.sh` declares failure. A smaller value fails faster but is more likely to time out on a slow first run. |
| `FORCE` | `0` or `1`. | `0` | `0` reuses the existing gallery bank when the local index files already exist. `1` ignores the existing gallery bank, re-encodes the gallery, and rewrites the search config. |
| `ENCODE_BATCH_SIZE` | Positive integer. | `8` | Controls gallery-encoding batch size. Larger values can improve throughput but use more memory. |

## Direct Deployment

These variables apply when `deploy_best.sh` is run directly.

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `GALLERY_PATH` | Absolute path to a local image folder. Required. | none | Selects the gallery to index and serve. |
| `CONFIG_OUTPUT` | Path to a JSON file. | `deployment/search_config_gallery_mlx.json` | Changes where the generated search config is written. |
| `METADATA_MANIFEST` | Path to a local JSONL file, or unset. | `datasets/private_gallery_local/full_manifest.jsonl` | If set, the runtime loads weak captions from that file for metadata display and the small metadata-based ranking boost. If unset, search still works, but the extra metadata boost and manifest-backed metadata display are disabled. |
| `HOST` | IP address or hostname string. Typical values: `127.0.0.1`, `0.0.0.0`. | `127.0.0.1` | `127.0.0.1` keeps the app reachable only from the local machine. `0.0.0.0` binds every interface and makes the app reachable from the local network. Any other address binds only that specific interface. |
| `PORT` | Unused TCP port number. | `36168` | Changes the web URL and the runtime log and PID filenames. |
| `MODEL_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | `bfloat16` is the normal fast path for gallery encoding and query encoding. `float32` uses more memory and runs slower, but is the more conservative option if you want full-precision inference. |
| `MODEL_WEIGHTS` | Absolute path to a `weights.safetensors` file, or unset. | latest local adaptation if present, otherwise Stage 1 | If unset, deployment prefers the latest local adaptation weights and falls back to Stage 1 when no local adaptation exists. If set, it uses the file you provided for both gallery encoding and query encoding. |
| `FORCE` | `0` or `1`. | `0` | `0` reuses the existing gallery bank when the local index files already exist. `1` ignores the existing gallery bank and reruns gallery encoding. |
| `ENCODE_BATCH_SIZE` | Positive integer. | `8` | Controls gallery-encoding batch size. Larger values can improve throughput but use more memory. |

## Data Preparation

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `PRIVATE_GALLERY_PATH` | Absolute path to a local image folder. | required unless manifests already exist | Builds `full_manifest.jsonl` and `private_adapt_data.jsonl` from the user's gallery. |
| `PREPARE_PUBLIC_DATA` | `0` or `1`. | `0` | `0` builds only the local gallery files used by Stage 2 adaptation. `1` also downloads and prepares Flickr30k plus Screen2Words for full Stage 1 retraining. |
| `FORCE` | `0` or `1`. | `0` | `0` reuses local public Stage 1 datasets when they already meet the row threshold. `1` refreshes those public datasets even if the local copies already exist. |

## Gallery-Specific Adaptation

These variables apply to `adapt_best.sh`. In the shipped default Stage 2 path, the text tower stays frozen.

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `FINAL_RUN_NAME` | Directory name under `logs/`. | `semanticgallery_private_data_adapted` | Changes where Stage 2 writes weights, summary files, and history. |
| `STAGE1_WEIGHTS` | Absolute path to a `weights.safetensors` file, or unset. | local Stage 1 if present, otherwise the published checkpoint | If unset, `adapt_best.sh` uses the local Stage 1 checkpoint when present and falls back to the published checkpoint otherwise. If set, it starts Stage 2 from the file you provided. |
| `MAX_EPOCHS_STAGE2` | Positive integer. | `1` | Controls how many Stage 2 epochs are run. |
| `TRAIN_BATCH_SIZE` | Positive integer. | `4` | Controls the public text-image batch size used during Stage 2. |
| `TRAIN_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | `bfloat16` is the normal fast path for Stage 2 and is the shipped default. `float32` uses more memory and runs slower, but is the more conservative option if you want full-precision training. |
| `VISION_UNFREEZE_LAST_N` | Integer `>= 0`. | `2` | Controls how many vision blocks remain trainable. `0` freezes the vision tower. Larger values tune more of the encoder and use more memory. |
| `LR_STAGE2` | Positive float. | `5e-6` | Controls the Stage 2 learning rate. Larger values adapt faster but increase drift risk. |
| `PRIVATE_BATCH_SIZE` | Positive integer. | `8` | Controls how many local images are used per private Stage 2 step. |
| `PRIVATE_REPEATS_PER_EPOCH` | Positive integer. | `2` | Controls how many passes Stage 2 makes over the capped local set in one epoch. |
| `PRIVATE_INSTANCE_WEIGHT` | Non-negative float. | `0.3` | Weights the image-instance loss. Larger values pull harder toward gallery-specific invariance. |
| `PRIVATE_DISTILL_WEIGHT` | Non-negative float. | `0.15` | Weights the distillation loss. Larger values keep the adapted image encoder closer to the Stage 1 teacher. |
| `MAX_TRAIN_STEPS` | Positive integer, or unset. | unset | If unset, Stage 2 runs every scheduled training step for the epoch. If set, it stops after that many training steps and is mainly useful for smoke tests or short experiments. |
| `MAX_VAL_STEPS` | Positive integer, or unset. | unset | If unset, Stage 2 runs the full validation pass. If set, it stops validation early and is mainly useful for smoke tests or short experiments. |

## Full Retraining

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `PUBLIC_RUN_NAME` | Directory name under `logs/`. | `semanticgallery_public_stage1` | Changes where Stage 1 writes weights, summary files, and history. |
| `FINAL_RUN_NAME` | Directory name under `logs/`. | `semanticgallery_public_plus_private_data` | Changes where the final Stage 1 -> Stage 2 run writes weights, summary files, and history. |
| `MAX_EPOCHS_STAGE1` | Positive integer. | `1` | Controls how many Stage 1 epochs are run. |
| `MAX_EPOCHS_STAGE2` | Positive integer. | `1` | Controls how many Stage 2 epochs run after Stage 1. |
| `TRAIN_BATCH_SIZE` | Positive integer. | `4` | Controls the public text-image batch size in both stages. |
| `TRAIN_PRECISION` | `bfloat16` or `float32`. | `bfloat16` | `bfloat16` is the normal fast path for both stages and is the shipped default. `float32` uses more memory and runs slower, but is the more conservative option if you want full-precision training. |
| `TEXT_UNFREEZE_LAST_N` | Integer `>= 0`. | `2` | Controls how many text blocks remain trainable during Stage 1. `0` freezes the text tower. Larger values tune more of the text encoder and use more memory. |
| `VISION_UNFREEZE_LAST_N` | Integer `>= 0`. | `2` | Controls how many vision blocks remain trainable in both stages. |
| `LR_STAGE1` | Positive float. | `1e-5` | Controls the Stage 1 learning rate. |
| `LR_STAGE2` | Positive float. | `5e-6` | Controls the Stage 2 learning rate after Stage 1. |
| `PRIVATE_BATCH_SIZE` | Positive integer. | `8` | Controls how many local images are used per private Stage 2 step. |
| `PRIVATE_REPEATS_PER_EPOCH` | Positive integer. | `2` | Controls how many passes Stage 2 makes over the capped local set in one epoch. |
| `PRIVATE_INSTANCE_WEIGHT` | Non-negative float. | `0.3` | Weights the image-instance loss in Stage 2. |
| `PRIVATE_DISTILL_WEIGHT` | Non-negative float. | `0.15` | Weights the distillation loss in Stage 2. |
| `MAX_TRAIN_STEPS` | Positive integer, or unset. | unset | If unset, each stage runs every scheduled training step for the epoch. If set, the current stage stops after that many training steps and is mainly useful for smoke tests or short experiments. |
| `MAX_VAL_STEPS` | Positive integer, or unset. | unset | If unset, each stage runs the full validation pass. If set, validation stops early and is mainly useful for smoke tests or short experiments. |

## Common Operations

Rebuild the gallery index after adding or changing files, or after rerunning gallery-specific adaptation with different weights:

```bash
FORCE=1 GALLERY_PATH=/absolute/path/to/gallery ./scripts/quickstart.sh
```

Stop the running service:

```bash
kill "$(cat logs/runtime/semanticgallery_36168.pid)"
```
