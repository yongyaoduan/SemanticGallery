#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${1:-${SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT:-$ROOT_DIR/.cache}}"

download_artifact_file() {
  local repo_type="$1"
  local repo_id="$2"
  local relative_path="$3"
  local filename="$4"
  local destination_dir="$ARTIFACT_ROOT/$relative_path"
  local destination_file="$destination_dir/$filename"
  local base_url="https://huggingface.co"

  if [[ "$repo_type" == "dataset" ]]; then
    base_url="https://huggingface.co/datasets"
  fi

  mkdir -p "$destination_dir"
  if [[ -s "$destination_file" ]]; then
    return
  fi

  curl -L --fail --progress-bar \
    -o "$destination_file" \
    "$base_url/$repo_id/resolve/main/$filename"
}

mkdir -p "$ARTIFACT_ROOT"

download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 config.json
download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 tokenizer.json
download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 tokenizer_config.json
download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 special_tokens_map.json
download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 preprocessor_config.json
download_artifact_file model Lucas20250626/semanticgallery-mlx-siglip2-stage1 semanticgallery/stage1 weights.safetensors
download_artifact_file model Lucas20250626/semanticgallery-mlx-siglip2-stage1 semanticgallery/stage1 summary.json
download_artifact_file dataset Lucas20250626/semanticgallery-stage2-public-anchor semanticgallery/stage2_public_anchor semanticgallery-stage2-public-anchor.tar.gz
download_artifact_file dataset Lucas20250626/semanticgallery-stage2-public-anchor semanticgallery/stage2_public_anchor sample_info.json

echo "$ARTIFACT_ROOT"
