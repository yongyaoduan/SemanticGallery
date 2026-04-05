# Benchmarks

This page records the selection result that led to the shipped MLX runtime on Apple Silicon. The benchmark harness and raw local artifacts are kept outside the Git repo; end users do not need them to run SemanticGallery.

## MPS vs MLX

`MPS` is PyTorch running on Apple's Metal backend. `MLX` is Apple's native array runtime for Apple Silicon. In this repo, the practical selection question is steady-state latency and low-precision stability on Apple hardware.

## Benchmark Setup

| Item | Value |
| --- | --- |
| Hardware | MacBook Air, Apple M4, 10 CPU cores, 32 GB unified memory |
| Software | Python `3.12.13`, `torch==2.5.1`, `transformers==4.57.6`, `mlx==0.31.1`, `mlx-embeddings==0.1.0` |
| Model | `google/siglip2-base-patch16-224` |
| Training workload | batch size `2`, `20` measured steps, `3` warmup steps |
| Trainable scope | last `2` text layers, last `2` vision layers, both projection heads, `logit_scale`, and `logit_bias` |
| Deployment workload | `100` measured queries, `2` warmup cycles |
| Deployment bank | the same prepared `10,410`-image gallery bank for both backends |
| Measurement mode | sequential single-request timing |
| Warmup purpose | exclude kernel compilation and cache fill from steady-state timing |

## Speed Results

| Stage | Backend | Precision | Mean | P95 |
| --- | --- | --- | --- | --- |
| Training | PyTorch + MPS | `float32` | `231.24 ms / step` | `250.78 ms` |
| Training | PyTorch + MPS | `float16` | `failed` | `failed` |
| Training | MLX | `float32` | `254.04 ms / step` | `261.80 ms` |
| Training | MLX | `bfloat16` | `165.20 ms / step` | `168.60 ms` |
| Deployment | PyTorch + MPS | `float32` | `12.64 ms / query` | `14.39 ms` |
| Deployment | PyTorch + MPS | `float16` | `19.92 ms / query` | `21.79 ms` |
| Deployment | MLX | `bfloat16` | `9.02 ms / query` | `10.35 ms` |

`failed` means the measured configuration did not stay numerically stable enough to complete the workload.

## Analysis

- Training: `MLX bfloat16` reduces mean step time from `231.24 ms` to `165.20 ms` relative to `PyTorch + MPS float32`, a `28.6%` latency reduction
- Training: `MLX float32` is slower than `PyTorch + MPS float32`; the win comes from stable `bfloat16`
- Training: `PyTorch + MPS float16` did not remain numerically stable on this workload
- Deployment: `MLX bfloat16` reduces mean query time from `12.64 ms` to `9.02 ms` relative to `PyTorch + MPS float32`, also a `28.6%` latency reduction
- Deployment: `MLX bfloat16` is `54.7%` faster than `PyTorch + MPS float16` on the same gallery bank

The shipped runtime is pure MLX because it gives the best low-precision performance on Apple Silicon while keeping the runtime stack simpler.

## Training Path Selection

`Base` is the released `google/siglip2-base-patch16-224` checkpoint with no repo-specific tuning. `Stage 1` fine-tunes that checkpoint on Flickr30k plus Screen2Words. `Stage 2 only` starts from the base checkpoint and runs only gallery-specific adaptation. `Stage 1 -> Stage 2` starts from the published Stage 1 checkpoint and then runs gallery-specific adaptation.

These numbers are reference selection results recorded during development. They document why the repo ships `Stage 1 -> Stage 2`; they are not reproduced by an in-repo benchmark command.

### Reference Setup

| Item | Value |
| --- | --- |
| Retrieval task | text-to-image retrieval |
| Base model | `google/siglip2-base-patch16-224` |
| Local adaptation data | up to `100` images from a personal gallery with phone photos and screenshots |
| Public held-out test | Flickr30k official test split: `1000` images, `5000` caption queries |
| Screenshot held-out test | Screen2Words official test split: `4310` images, `21550` caption queries |
| Metrics | `Recall@1`, `Recall@5`, `Recall@10` |

### Flickr30k Test

| Model | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Base | `0.7630` | `0.9240` | `0.9564` |
| Stage 1 only | `0.7864` | `0.9410` | `0.9672` |
| Stage 2 only | `0.7646` | `0.9246` | `0.9568` |
| Stage 1 -> Stage 2 | `0.7870` | `0.9408` | `0.9670` |

### Screen2Words Test

| Model | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Base | `0.1185` | `0.2538` | `0.3254` |
| Stage 1 only | `0.1570` | `0.3210` | `0.3989` |
| Stage 2 only | `0.1187` | `0.2553` | `0.3271` |
| Stage 1 -> Stage 2 | `0.1574` | `0.3211` | `0.3992` |

### Analysis

- `Stage 2 only` does not replace public fine-tuning
- `Stage 1` provides nearly all held-out retrieval gain
- `Stage 1 -> Stage 2` keeps those public gains while adding gallery-specific adaptation

The shipped training path is `Stage 1 -> Stage 2`.

## Stage 2 Weight Selection

The shipped Stage 2 loss keeps three terms active:

`L = 1.0 * L_public_txtimg + w_instance * L_private_instance + w_distill * L_distill`

The selection question is how much gallery-specific pull to add without drifting too far from the published Stage 1 checkpoint. The reference sweep below used the same Stage 1 checkpoint, the same `1000`-row public reference set, the same capped local adaptation set, the same seed, and one Stage 2 epoch.

### Reference Setup

| Item | Value |
| --- | --- |
| Base checkpoint | published Stage 1 checkpoint |
| Public Stage 2 reference set | `1000` rows: `500` Flickr30k + `500` Screen2Words |
| Local adaptation set | `32` demo-gallery images |
| Precision | `bfloat16` |
| Stage 2 epoch count | `1` |
| Public batch size | `4` |
| Private batch size | `8` |
| Metric | Stage 2 train loss and validation loss |

### Weight Sweep

| `w_instance` | `w_distill` | Train loss | Val loss |
| ---: | ---: | ---: | ---: |
| `0.30` | `0.15` | `0.6296` | `0.4931` |
| `0.40` | `0.08` | `0.6568` | `0.4937` |
| `0.50` | `0.10` | `0.6850` | `0.4935` |

### Analysis

- All three settings land in a narrow band, so Stage 2 is not highly sensitive to small weight changes in this range
- `0.30 / 0.15` produced the lowest validation loss in the reference sweep
- The shipped Stage 2 defaults use `w_instance=0.30` and `w_distill=0.15`
