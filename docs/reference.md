# Reference

This page lists the user-facing environment variables exposed by the shell entrypoints in `scripts/`.

## Quick Start

| Variable | Meaning | Default |
| --- | --- | --- |
| `GALLERY_PATH` | Folder to index and search | required |
| `STAGE1_WEIGHTS` | Optional published-checkpoint override for the adaptation step | local Stage 1 if present, otherwise the published checkpoint |
| `HOST` | Web app bind address | `127.0.0.1` |
| `PORT` | Web app port | `36168` |
| `CONFIG_OUTPUT` | Search-config output path for the current run | `deployment/search_config_gallery_mlx.json` |
| `METADATA_MANIFEST` | Manifest used for runtime metadata boost and metadata display | `datasets/private_gallery_local/full_manifest.jsonl` |
| `MODEL_PRECISION` | Precision used for gallery encoding and search | `bfloat16` |
| `RUNTIME_DIR` | Directory that stores runtime logs and PID files | `logs/runtime` |
| `LOG_FILE` | Startup log path | `logs/runtime/semanticgallery_<port>.log` |
| `PID_FILE` | PID file path | `logs/runtime/semanticgallery_<port>.pid` |
| `STARTUP_TIMEOUT_SECONDS` | Max wait time before startup is considered failed | `300` |
| `FORCE` | Rebuild the gallery index instead of reusing the existing one | `0` |
| `ENCODE_BATCH_SIZE` | Batch size for offline gallery encoding | `8` |

## Direct Deployment

These variables apply when `deploy_best.sh` is run directly.

| Variable | Meaning | Default |
| --- | --- | --- |
| `GALLERY_PATH` | Folder to index and search | required |
| `CONFIG_OUTPUT` | Search-config output path for the current run | `deployment/search_config_gallery_mlx.json` |
| `METADATA_MANIFEST` | Manifest used for runtime metadata boost and metadata display | `datasets/private_gallery_local/full_manifest.jsonl` |
| `HOST` | Web app bind address | `127.0.0.1` |
| `PORT` | Web app port | `36168` |
| `MODEL_PRECISION` | Precision used for gallery encoding and search | `bfloat16` |
| `MODEL_WEIGHTS` | Optional model-weight override used for gallery encoding and query encoding | latest local adaptation if present, otherwise Stage 1 |
| `FORCE` | Rebuild the gallery index instead of reusing the existing one | `0` |
| `ENCODE_BATCH_SIZE` | Batch size for offline gallery encoding | `8` |

## Data Preparation

| Variable | Meaning | Default |
| --- | --- | --- |
| `PRIVATE_GALLERY_PATH` | Gallery path used to build local manifests | required unless manifests already exist |
| `ALIASES_JSON` | Optional folder-alias JSON used during manifest generation | `datasets/private_gallery_local/aliases.local.json` if present |
| `QUERY_SUITE_PATH` | Optional local query suite used to prioritize folders during capped local sampling | `datasets/private_gallery_local/query_suite.local.json` if present |
| `PREPARE_PUBLIC_DATA` | Download and prepare the full public corpus for Stage 1 reproduction | `0` |
| `FORCE` | Refresh public datasets even if local copies already meet the minimum row threshold | `0` |

## Gallery-Specific Adaptation

These variables apply to `adapt_best.sh`. In the shipped default Stage 2 path, the text tower is frozen, so `TEXT_UNFREEZE_LAST_N` does not change the trainable set unless you also change the script behavior.

| Variable | Meaning | Default |
| --- | --- | --- |
| `FINAL_RUN_NAME` | Output directory name under `logs/` | `semanticgallery_private_data_adapted` |
| `STAGE1_WEIGHTS` | Optional starting checkpoint for adaptation | local Stage 1 if present, otherwise the published checkpoint |
| `MAX_EPOCHS_STAGE2` | Stage 2 epoch count | `1` |
| `TRAIN_BATCH_SIZE` | Public text-image batch size | `4` |
| `TRAIN_PRECISION` | Training precision | `bfloat16` |
| `VISION_UNFREEZE_LAST_N` | Number of vision blocks left trainable | `2` |
| `LR_STAGE2` | Stage 2 learning rate | `5e-6` |
| `PRIVATE_BATCH_SIZE` | Per-step private image batch size | `8` |
| `PRIVATE_REPEATS_PER_EPOCH` | Number of passes over the capped local set per Stage 2 epoch | `2` |
| `PRIVATE_INSTANCE_WEIGHT` | Weight of the image-instance loss | `0.5` |
| `PRIVATE_DISTILL_WEIGHT` | Weight of the distillation loss | `0.1` |
| `MAX_TRAIN_STEPS` | Optional training-step cap for smoke tests | unset |
| `MAX_VAL_STEPS` | Optional validation-step cap for smoke tests | unset |

## Full Retraining

| Variable | Meaning | Default |
| --- | --- | --- |
| `PUBLIC_RUN_NAME` | Stage 1 output directory name under `logs/` | `semanticgallery_public_stage1` |
| `FINAL_RUN_NAME` | Final Stage 1 -> Stage 2 output directory name under `logs/` | `semanticgallery_public_plus_private_data` |
| `MAX_EPOCHS_STAGE1` | Stage 1 epoch count | `1` |
| `MAX_EPOCHS_STAGE2` | Stage 2 epoch count after Stage 1 | `1` |
| `TRAIN_BATCH_SIZE` | Public text-image batch size | `4` |
| `TRAIN_PRECISION` | Training precision | `bfloat16` |
| `TEXT_UNFREEZE_LAST_N` | Number of text blocks left trainable during Stage 1 | `2` |
| `VISION_UNFREEZE_LAST_N` | Number of vision blocks left trainable | `2` |
| `LR_STAGE1` | Stage 1 learning rate | `1e-5` |
| `LR_STAGE2` | Stage 2 learning rate | `5e-6` |
| `PRIVATE_BATCH_SIZE` | Per-step private image batch size during Stage 2 | `8` |
| `PRIVATE_REPEATS_PER_EPOCH` | Number of passes over the capped local set per Stage 2 epoch | `2` |
| `PRIVATE_INSTANCE_WEIGHT` | Weight of the image-instance loss | `0.5` |
| `PRIVATE_DISTILL_WEIGHT` | Weight of the distillation loss | `0.1` |
| `MAX_TRAIN_STEPS` | Optional training-step cap for smoke tests | unset |
| `MAX_VAL_STEPS` | Optional validation-step cap for smoke tests | unset |

## Common Operations

Rebuild the gallery index after adding or changing files, or after rerunning gallery-specific adaptation with different weights:

```bash
FORCE=1 GALLERY_PATH=/absolute/path/to/gallery ./scripts/quickstart.sh
```

Stop the running service:

```bash
kill "$(cat logs/runtime/semanticgallery_36168.pid)"
```
