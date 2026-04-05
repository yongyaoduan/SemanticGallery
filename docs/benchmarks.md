# Benchmarks

SemanticGallery ships the MLX runtime on Apple Silicon. `PyTorch + MPS` is kept only as a reference baseline for selection.

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

## Analysis

- Training: `MLX bfloat16` reduces mean step time from `231.24 ms` to `165.20 ms` relative to `PyTorch + MPS float32`, a `28.6%` latency reduction
- Training: `MLX float32` is slower than `PyTorch + MPS float32`; the win comes from stable `bfloat16`
- Training: `PyTorch + MPS float16` did not remain numerically stable on this workload
- Deployment: `MLX bfloat16` reduces mean query time from `12.64 ms` to `9.02 ms` relative to `PyTorch + MPS float32`, also a `28.6%` latency reduction
- Deployment: `MLX bfloat16` is `54.7%` faster than `PyTorch + MPS float16` on the same gallery bank

The shipped runtime is pure MLX because it gives the best low-precision performance on Apple Silicon while keeping the runtime stack simpler.
