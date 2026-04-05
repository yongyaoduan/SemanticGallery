# Reference

Set variables inline before the command:

```bash
NAME=value OTHER=value ./scripts/quickstart.sh
```

This page lists the user-facing environment variables exposed by the shell entrypoints in `scripts/`.

## Quick Start

| Variable | Details |
| --- | --- |
| `GALLERY_DIR` | Required. Set an absolute path to the local image folder that SemanticGallery should adapt, index, and serve. |
| `STAGE1_WEIGHTS_FILE_PATH` | Default: `logs/semanticgallery_public_stage1/weights.safetensors`. Stage 1 model file path. SemanticGallery checks this path first. If a local file exists there, it uses that file. Otherwise it downloads the published Stage 1 checkpoint and uses that. |
| `HOST` | Default: `127.0.0.1`. `127.0.0.1`: bind only the local machine. `0.0.0.0`: bind every interface and expose the app to the local network. Any other address: bind only that specific interface. |
| `PORT` | Default: `36168`. Set any unused TCP port to change the web URL and the runtime log and PID filenames. |
| `CONFIG_FILE_PATH` | Default: `deployment/search_config_gallery_mlx.json`. Set a JSON file path to change where the generated search config is written. |
| `METADATA_MANIFEST_FILE_PATH` | Default: `datasets/private_gallery_local/full_manifest.jsonl`. Metadata manifest file path. SemanticGallery reads weak captions from this JSONL file for metadata display and the small metadata-based ranking boost. |
| `MODEL_PRECISION` | Default: `bfloat16`. `bfloat16`: normal fast path for gallery encoding and query encoding. `float32`: slower and uses more memory, but is the conservative full-precision option. |
| `RUNTIME_DIR` | Default: `logs/runtime`. Set a directory path to change where `quickstart.sh` writes the startup log and PID file. |
| `LOG_FILE_PATH` | Default: `logs/runtime/semanticgallery_<port>.log`. Set a file path to override the startup log location. |
| `PID_FILE_PATH` | Default: `logs/runtime/semanticgallery_<port>.pid`. Set a file path to override the PID file location. |
| `STARTUP_TIMEOUT_SECONDS` | Default: `300`. Set a larger integer to give adaptation and startup more time before failure. Set a smaller integer to fail faster. |
| `FORCE` | Default: `0`. `0`: reuse the existing gallery bank when local index files already exist. `1`: ignore the existing gallery bank, re-encode the gallery, and rewrite the search config. |
| `ENCODE_BATCH_SIZE` | Default: `8`. Set a larger positive integer to improve throughput at the cost of higher memory use. Set a smaller value to reduce memory use. |

## Direct Deployment

These variables apply when `deploy_best.sh` is run directly.

| Variable | Details |
| --- | --- |
| `GALLERY_DIR` | Required. Set an absolute path to the local image folder that should be indexed and served. |
| `CONFIG_FILE_PATH` | Default: `deployment/search_config_gallery_mlx.json`. Set a JSON file path to change where the generated search config is written. |
| `METADATA_MANIFEST_FILE_PATH` | Default: `datasets/private_gallery_local/full_manifest.jsonl`. Metadata manifest file path. SemanticGallery reads weak captions from this JSONL file for metadata display and the small metadata-based ranking boost. |
| `HOST` | Default: `127.0.0.1`. `127.0.0.1`: bind only the local machine. `0.0.0.0`: bind every interface and expose the app to the local network. Any other address: bind only that specific interface. |
| `PORT` | Default: `36168`. Set any unused TCP port to change the web URL and the runtime log and PID filenames. |
| `MODEL_PRECISION` | Default: `bfloat16`. `bfloat16`: normal fast path for gallery encoding and query encoding. `float32`: slower and uses more memory, but is the conservative full-precision option. |
| `MODEL_WEIGHTS_FILE_PATH` | Default: `logs/semanticgallery_private_data_adapted/weights.safetensors`. Deployment model file path. SemanticGallery checks this path first. If a local file exists there, it uses that file. Otherwise it falls back to the published Stage 1 checkpoint and uses that. |
| `FORCE` | Default: `0`. `0`: reuse the existing gallery bank when local index files already exist. `1`: ignore the existing gallery bank and rerun gallery encoding. |
| `ENCODE_BATCH_SIZE` | Default: `8`. Set a larger positive integer to improve throughput at the cost of higher memory use. Set a smaller value to reduce memory use. |

## Data Preparation

| Variable | Details |
| --- | --- |
| `PRIVATE_GALLERY_DIR` | Required unless the local JSONL files already exist. Set an absolute path to the user's gallery to build `full_manifest.jsonl` and `private_adapt_data.jsonl`. |
| `PREPARE_PUBLIC_DATA` | Default: `0`. `0`: build only the local gallery files used by Stage 2 adaptation. `1`: also download and prepare Flickr30k plus Screen2Words for full Stage 1 retraining. |
| `FORCE` | Default: `0`. `0`: reuse local public Stage 1 datasets when they already meet the row threshold. `1`: refresh those public datasets even if the local copies already exist. |

## Gallery-Specific Adaptation

These variables apply to `adapt_best.sh`. In the shipped default Stage 2 path, the text tower stays frozen.

| Variable | Details |
| --- | --- |
| `FINAL_RUN_NAME` | Default: `semanticgallery_private_data_adapted`. Set a directory name under `logs/` to change where Stage 2 writes weights, summary files, and history. |
| `STAGE1_WEIGHTS_FILE_PATH` | Default: `logs/semanticgallery_public_stage1/weights.safetensors`. Stage 1 model file path. Stage 2 checks this path first. If a local file exists there, it uses that file. Otherwise it downloads the published Stage 1 checkpoint and uses that. |
| `MAX_EPOCHS_STAGE2` | Default: `1`. Set a larger positive integer to run more Stage 2 epochs. |
| `TRAIN_BATCH_SIZE` | Default: `4`. Set a larger positive integer to increase the public text-image batch size used during Stage 2. |
| `TRAIN_PRECISION` | Default: `bfloat16`. `bfloat16`: normal fast path for Stage 2 and the shipped default. `float32`: slower and uses more memory, but is the conservative full-precision option. |
| `VISION_UNFREEZE_LAST_N` | Default: `2`. `0`: freeze the vision tower. Any integer `>= 1`: keep that many final vision blocks trainable. Larger values tune more of the encoder and use more memory. |
| `LR_STAGE2` | Default: `5e-6`. Set a larger positive float to adapt faster at the cost of higher drift risk. Set a smaller value for more conservative updates. |
| `PRIVATE_BATCH_SIZE` | Default: `8`. Set a larger positive integer to use more local images per private Stage 2 step. |
| `PRIVATE_REPEATS_PER_EPOCH` | Default: `2`. Set a larger positive integer to make more passes over the capped local set in one epoch. |
| `PRIVATE_INSTANCE_WEIGHT` | Default: `0.3`. Set a non-negative float. Larger values pull harder toward gallery-specific invariance. `0` disables the instance loss. |
| `PRIVATE_DISTILL_WEIGHT` | Default: `0.15`. Set a non-negative float. Larger values keep the adapted image encoder closer to the Stage 1 teacher. `0` disables the distillation loss. |
| `MAX_TRAIN_STEPS` | Default: unset. Unset: run every scheduled training step for the epoch. Set a positive integer: stop after that many training steps. |
| `MAX_VAL_STEPS` | Default: unset. Unset: run the full validation pass. Set a positive integer: stop validation early. |

## Full Retraining

| Variable | Details |
| --- | --- |
| `PUBLIC_RUN_NAME` | Default: `semanticgallery_public_stage1`. Set a directory name under `logs/` to change where Stage 1 writes weights, summary files, and history. |
| `FINAL_RUN_NAME` | Default: `semanticgallery_public_plus_private_data`. Set a directory name under `logs/` to change where the final Stage 1 -> Stage 2 run writes weights, summary files, and history. |
| `MAX_EPOCHS_STAGE1` | Default: `1`. Set a larger positive integer to run more Stage 1 epochs. |
| `MAX_EPOCHS_STAGE2` | Default: `1`. Set a larger positive integer to run more Stage 2 epochs after Stage 1. |
| `TRAIN_BATCH_SIZE` | Default: `4`. Set a larger positive integer to increase the public text-image batch size in both stages. |
| `TRAIN_PRECISION` | Default: `bfloat16`. `bfloat16`: normal fast path for both stages and the shipped default. `float32`: slower and uses more memory, but is the conservative full-precision option. |
| `TEXT_UNFREEZE_LAST_N` | Default: `2`. `0`: freeze the text tower during Stage 1. Any integer `>= 1`: keep that many final text blocks trainable. Larger values tune more of the text encoder and use more memory. |
| `VISION_UNFREEZE_LAST_N` | Default: `2`. `0`: freeze the vision tower. Any integer `>= 1`: keep that many final vision blocks trainable. Larger values tune more of the encoder and use more memory. |
| `LR_STAGE1` | Default: `1e-5`. Set a larger positive float to adapt faster at the cost of higher drift risk. Set a smaller value for more conservative updates. |
| `LR_STAGE2` | Default: `5e-6`. Set a larger positive float to adapt faster at the cost of higher drift risk. Set a smaller value for more conservative updates. |
| `PRIVATE_BATCH_SIZE` | Default: `8`. Set a larger positive integer to use more local images per private Stage 2 step. |
| `PRIVATE_REPEATS_PER_EPOCH` | Default: `2`. Set a larger positive integer to make more passes over the capped local set in one epoch. |
| `PRIVATE_INSTANCE_WEIGHT` | Default: `0.3`. Set a non-negative float. Larger values pull harder toward gallery-specific invariance. `0` disables the instance loss. |
| `PRIVATE_DISTILL_WEIGHT` | Default: `0.15`. Set a non-negative float. Larger values keep the adapted image encoder closer to the Stage 1 teacher. `0` disables the distillation loss. |
| `MAX_TRAIN_STEPS` | Default: unset. Unset: each stage runs every scheduled training step for the epoch. Set a positive integer: stop the current stage after that many training steps. |
| `MAX_VAL_STEPS` | Default: unset. Unset: each stage runs the full validation pass. Set a positive integer: stop validation early. |

## Common Operations

Rebuild the gallery index after adding or changing files, or after rerunning gallery-specific adaptation with different weights:

```bash
FORCE=1 GALLERY_DIR=/absolute/path/to/gallery ./scripts/quickstart.sh
```

Stop the running service:

```bash
kill "$(cat logs/runtime/semanticgallery_36168.pid)"
```
