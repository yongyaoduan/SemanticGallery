# Data and Training

SemanticGallery has two training paths.

## Default Path: Local Adaptation

Use this path in normal operation. It reuses the published Stage 1 checkpoint, downloads the compact public anchor, and adapts only on the target gallery.

```bash
PRIVATE_GALLERY_PATH=/absolute/path/to/gallery ./scripts/prepare_data.sh
./scripts/adapt_best.sh
```

This path writes the adapted weights to `logs/semanticgallery_private_data_adapted/weights.safetensors`.

### Data Used by the Default Path

- Published Stage 1 checkpoint: [Lucas20250626/semanticgallery-mlx-siglip2-stage1](https://huggingface.co/Lucas20250626/semanticgallery-mlx-siglip2-stage1)
- Stage 2 public anchor: [Lucas20250626/semanticgallery-stage2-public-anchor](https://huggingface.co/datasets/Lucas20250626/semanticgallery-stage2-public-anchor)
- Local adaptation set: up to `100` images sampled from the target gallery

The local adaptation set comes from the user's own gallery and mixes phone photos and screenshots. Stage 2 does not optimize on private captions. It keeps the text tower frozen and uses image-only adaptation on the local set while continuing to see public text-image batches from the public anchor.

The default adaptation loss is:

`L = 1.0 * L_public_txtimg + 0.5 * L_private_instance + 0.1 * L_distill`

- `L_public_txtimg` keeps the published retrieval space active
- `L_private_instance` pulls two augmented views of the same local image together
- `L_distill` keeps the adapted image encoder close to the published Stage 1 teacher

## Full Retraining

Use this path only when you want to reproduce or replace the published Stage 1 checkpoint.

```bash
PREPARE_PUBLIC_DATA=1 PRIVATE_GALLERY_PATH=/absolute/path/to/gallery ./scripts/prepare_data.sh
./scripts/train_best.sh
```

The reference public-training run took about `42` minutes on an Apple M4 MacBook Air. Reserve roughly one hour when reproducing it locally.

### Public Training Corpus

| Source | Records | Share of public corpus |
| --- | ---: | ---: |
| Flickr30k | `31,783` | `63.7%` |
| Screen2Words train | `15,743` | `31.6%` |
| Screen2Words val | `2,364` | `4.7%` |
| Total | `49,890` | `100%` |

### Why Stage 1 -> Stage 2

`Base` is the released `google/siglip2-base-patch16-224` checkpoint with no repo-specific tuning. `Stage 1` fine-tunes that checkpoint on Flickr30k plus Screen2Words. `Stage 2` starts from the published Stage 1 weights and applies local caption-free adaptation.

#### Reference Setup

| Item | Value |
| --- | --- |
| Retrieval task | text-to-image retrieval |
| Base model | `google/siglip2-base-patch16-224` |
| Base | pretrained checkpoint only |
| Stage 1 | `1` public fine-tuning epoch on Flickr30k + Screen2Words |
| Stage 2 only | caption-free adaptation from the base checkpoint |
| Stage 1 -> Stage 2 | published Stage 1 followed by local caption-free adaptation |
| Local adaptation data | up to `100` images from a personal gallery with phone photos and screenshots |
| Public held-out tests | Flickr30k official test split: `1000` images, `5000` caption queries |
| Screenshot held-out tests | Screen2Words official test split: `4310` images, `21550` caption queries |
| Metrics | `Recall@1`, `Recall@5`, `Recall@10` |

#### Reference Results

**Flickr30k test**

| Model | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Base | `0.7630` | `0.9240` | `0.9564` |
| Stage 1 only | `0.7864` | `0.9410` | `0.9672` |
| Stage 2 only | `0.7646` | `0.9246` | `0.9568` |
| Stage 1 -> Stage 2 | `0.7870` | `0.9408` | `0.9670` |

**Screen2Words test**

| Model | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Base | `0.1185` | `0.2538` | `0.3254` |
| Stage 1 only | `0.1570` | `0.3210` | `0.3989` |
| Stage 2 only | `0.1187` | `0.2553` | `0.3271` |
| Stage 1 -> Stage 2 | `0.1574` | `0.3211` | `0.3992` |

#### Analysis

- `Stage 2 only` does not replace public fine-tuning
- `Stage 1` provides nearly all held-out retrieval gain
- `Stage 1 -> Stage 2` preserves those public gains while adding local gallery adaptation

The selected training path is `Stage 1 -> Stage 2`.
