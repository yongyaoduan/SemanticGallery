# Training

SemanticGallery has two training paths:

- the default path used by `quickstart.sh`, which reuses the published Stage 1 checkpoint and adapts it to one gallery
- the full path, which reruns public Stage 1 training and then still uses Stage 2 for gallery-specific adaptation

## Default Path: Gallery-Specific Adaptation

```bash
PRIVATE_GALLERY_PATH=/absolute/path/to/gallery ./scripts/prepare_data.sh
./scripts/adapt_best.sh
```

This is the normal path. It writes adapted weights to `logs/semanticgallery_private_data_adapted/weights.safetensors`.

### Inputs

| Input | Role |
| --- | --- |
| Published Stage 1 checkpoint | Starting point for the image and text encoders |
| Stage 2 public reference set | Keeps public text-image alignment active during Stage 2 |
| Local adaptation set | Up to `100` images sampled from the target gallery |

The local adaptation set comes from the user's own gallery and typically mixes phone photos and screenshots.

### What Stage 2 Optimizes

Stage 2 keeps the text tower frozen and combines three losses:

`L = 1.0 * L_public_txtimg + 0.3 * L_private_instance + 0.15 * L_distill`

| Loss | Purpose |
| --- | --- |
| `L_public_txtimg` | Keeps the published retrieval space aligned to public text-image supervision |
| `L_private_instance` | Pulls two augmented views of the same local image together |
| `L_distill` | Keeps the adapted image encoder close to the published Stage 1 teacher |

The shipped weights were selected from a small Stage 2 sweep on the same published Stage 1 checkpoint, the same `1000`-row public reference set, and the same capped local adaptation set. Three candidates were tested with the same seed and one Stage 2 epoch:

| `L_private_instance` | `L_distill` | Train loss | Val loss |
| ---: | ---: | ---: | ---: |
| `0.30` | `0.15` | `0.6296` | `0.4931` |
| `0.40` | `0.08` | `0.6568` | `0.4937` |
| `0.50` | `0.10` | `0.6850` | `0.4935` |

The differences are small, but `0.30 / 0.15` gave the lowest validation loss in that sweep. That is the default shipped Stage 2 setting.

## Full Retraining

```bash
PREPARE_PUBLIC_DATA=1 PRIVATE_GALLERY_PATH=/absolute/path/to/gallery ./scripts/prepare_data.sh
./scripts/train_best.sh
```

Use this path only when you want to reproduce or replace the published Stage 1 checkpoint.

The reference Stage 1 run took about `42` minutes on an Apple M4 MacBook Air. In practice, reserve roughly one hour when reproducing it locally.

### Public Training Corpus

| Source | Records | Share of public corpus |
| --- | ---: | ---: |
| Flickr30k | `31,783` | `63.7%` |
| Screen2Words train | `15,743` | `31.6%` |
| Screen2Words val | `2,364` | `4.7%` |
| Total | `49,890` | `100%` |

The held-out comparison that selected `Stage 1 -> Stage 2` is documented in [benchmarks.md](benchmarks.md).
