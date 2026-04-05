#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_PATH="$ROOT_DIR/.venv/bin/python"
MLX_MODEL_DIR="$ROOT_DIR/.cache/mlx/siglip2-base-patch16-224-f32"
LOCAL_STAGE1_WEIGHTS_FILE_PATH="$ROOT_DIR/logs/semanticgallery_public_stage1/weights.safetensors"
LOCAL_FINAL_WEIGHTS_FILE_PATH="$ROOT_DIR/logs/semanticgallery_private_data_adapted/weights.safetensors"
PUBLISHED_STAGE1_REPO_ID="Lucas20250626/semanticgallery-mlx-siglip2-stage1"
PUBLISHED_STAGE1_REVISION="main"
PUBLISHED_STAGE1_CACHE_DIR="$ROOT_DIR/.cache/semanticgallery/stage1"
PUBLISHED_STAGE1_WEIGHTS_FILE_PATH="$PUBLISHED_STAGE1_CACHE_DIR/weights.safetensors"
PUBLISHED_STAGE1_SUMMARY_FILE_PATH="$PUBLISHED_STAGE1_CACHE_DIR/summary.json"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_REPO_ID="Lucas20250626/semanticgallery-stage2-public-anchor"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_REVISION="main"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR="$ROOT_DIR/.cache/semanticgallery/stage2_public_anchor"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_ARCHIVE_FILE_PATH="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR/semanticgallery-stage2-public-anchor.tar.gz"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_METADATA_FILE_PATH="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR/sample_info.json"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT_DIR="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR/extracted"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT_DIR/flickr30k"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST_FILE_PATH="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT_DIR/screen2words/manifest.jsonl"

timestamp() {
  date '+%H:%M:%S'
}

log_step() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >&2
}

log_kv() {
  printf '  - %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required."
}

ensure_env() {
  require_uv
  if [[ ! -x "$PYTHON_BIN_PATH" ]]; then
    log_step "Creating Python environment"
    uv venv "$ROOT_DIR/.venv" --python 3.12
  fi
  if ! "$PYTHON_BIN_PATH" - <<'PY' >/dev/null 2>&1; then
import datasets, fastapi, huggingface_hub, jinja2, mlx, mlx_embeddings, pillow_heif, tqdm, uvicorn
PY
    log_step "Installing Python dependencies"
    uv pip install --python "$PYTHON_BIN_PATH" -r "$ROOT_DIR/requirements.txt"
  fi
}

ensure_mlx_model() {
  ensure_env
  if [[ ! -f "$MLX_MODEL_DIR/config.json" ]]; then
    log_step "Preparing MLX SigLIP2 model cache"
    "$PYTHON_BIN_PATH" - <<PY
from pathlib import Path
from mlx_embeddings.convert import convert

root = Path("${ROOT_DIR}")
model_dir = root / ".cache" / "mlx" / "siglip2-base-patch16-224-f32"
model_dir.parent.mkdir(parents=True, exist_ok=True)
convert(
    "google/siglip2-base-patch16-224",
    mlx_path=model_dir.as_posix(),
    dtype="float32",
    skip_vision=False,
)
PY
  fi
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "missing directory: $path"
}

ensure_port_free() {
  local host="$1"
  local port="$2"
  if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $port is already in use on $host"
  fi
}

ensure_published_stage1_weights() {
  ensure_env
  if [[ ! -f "$PUBLISHED_STAGE1_WEIGHTS_FILE_PATH" ]]; then
    log_step "Downloading published Stage 1 weights"
    log_kv "repo_id=$PUBLISHED_STAGE1_REPO_ID"
    log_kv "files=weights.safetensors,summary.json"
    mkdir -p "$PUBLISHED_STAGE1_CACHE_DIR"
    "$PYTHON_BIN_PATH" - <<PY
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

cache_dir = Path("${PUBLISHED_STAGE1_CACHE_DIR}")
cache_dir.mkdir(parents=True, exist_ok=True)
for filename in ("weights.safetensors", "summary.json"):
    print(f"downloading_stage1_file={filename}", file=sys.stderr)
    hf_hub_download(
        repo_id="${PUBLISHED_STAGE1_REPO_ID}",
        repo_type="model",
        revision="${PUBLISHED_STAGE1_REVISION}",
        filename=filename,
        local_dir=cache_dir.as_posix(),
    )
PY
  fi
}

ensure_published_stage2_public_anchor() {
  ensure_env
  if [[ -f "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR/captions.txt" && -f "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST_FILE_PATH" ]]; then
    log_step "Reusing cached Stage 2 public anchor"
    log_kv "cache_dir=$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR"
    return
  fi

  log_step "Downloading Stage 2 public anchor"
  log_kv "repo_id=$PUBLISHED_STAGE2_PUBLIC_ANCHOR_REPO_ID"
  log_kv "files=semanticgallery-stage2-public-anchor.tar.gz,sample_info.json"
  mkdir -p "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR"
  "$PYTHON_BIN_PATH" - <<PY
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

cache_dir = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR}")
archive_path = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_ARCHIVE_FILE_PATH}")
metadata_path = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_METADATA_FILE_PATH}")
extract_root = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT_DIR}")
tmp_root = cache_dir / "extracting"

cache_dir.mkdir(parents=True, exist_ok=True)
for filename in ("semanticgallery-stage2-public-anchor.tar.gz", "sample_info.json"):
    print(f"downloading_stage2_public_anchor_file={filename}", file=sys.stderr)
    hf_hub_download(
        repo_id="${PUBLISHED_STAGE2_PUBLIC_ANCHOR_REPO_ID}",
        repo_type="dataset",
        revision="${PUBLISHED_STAGE2_PUBLIC_ANCHOR_REVISION}",
        filename=filename,
        local_dir=cache_dir.as_posix(),
    )

if tmp_root.exists():
    shutil.rmtree(tmp_root)
tmp_root.mkdir(parents=True, exist_ok=True)
print("extracting_stage2_public_anchor=true", file=sys.stderr)
with tarfile.open(archive_path, "r:gz") as tar:
    try:
        tar.extractall(tmp_root, filter="data")
    except TypeError:
        tar.extractall(tmp_root)

if extract_root.exists():
    shutil.rmtree(extract_root)
tmp_root.rename(extract_root)

if metadata_path.exists():
    shutil.copy2(metadata_path, extract_root / "sample_info.json")
print("stage2_public_anchor_ready=true", file=sys.stderr)
PY
}

resolve_stage1_weights_file() {
  local preferred_file_path="${1:-$LOCAL_STAGE1_WEIGHTS_FILE_PATH}"
  if [[ -f "$preferred_file_path" ]]; then
    printf '%s\n' "$preferred_file_path"
    return
  fi
  ensure_published_stage1_weights
  printf '%s\n' "$PUBLISHED_STAGE1_WEIGHTS_FILE_PATH"
}

resolve_weights_file() {
  local preferred_file_path="${1:-$LOCAL_FINAL_WEIGHTS_FILE_PATH}"
  if [[ -f "$preferred_file_path" ]]; then
    printf '%s\n' "$preferred_file_path"
    return
  fi
  printf '%s\n' "$(resolve_stage1_weights_file "${STAGE1_WEIGHTS_FILE_PATH:-$LOCAL_STAGE1_WEIGHTS_FILE_PATH}")"
}

prepare_mlx_search_config() {
  local gallery_dir="$1"
  local config_file_path="$2"
  local metadata_manifest_file_path="${3:-}"
  local weights_file_path="${4:-}"
  local precision="${5:-bfloat16}"
  local gallery_name
  local embeddings_file_path
  local indexed_paths_file_path
  local skipped_images_file_path
  local legacy_embeddings_file_path
  local legacy_indexed_paths_file_path
  local legacy_skipped_images_file_path
  local encode_cmd
  local create_cmd

  ensure_mlx_model

  gallery_dir="$(cd "$gallery_dir" && pwd)"
  gallery_name="$(basename "$gallery_dir")"
  embeddings_file_path="$ROOT_DIR/deployment/${gallery_name}_mlx_siglip2_embeddings.npy"
  indexed_paths_file_path="$ROOT_DIR/deployment/${gallery_name}_mlx_siglip2.paths.txt"
  skipped_images_file_path="$ROOT_DIR/deployment/${gallery_name}_mlx_siglip2_skipped.json"
  legacy_embeddings_file_path="$ROOT_DIR/deployment/${gallery_name}_siglip2_embeddings.npy"
  legacy_indexed_paths_file_path="$ROOT_DIR/deployment/${gallery_name}_siglip2.paths.txt"
  legacy_skipped_images_file_path="$ROOT_DIR/deployment/${gallery_name}_siglip2_skipped.json"

  if [[ "${FORCE:-0}" != "1" && -z "$weights_file_path" && -f "$legacy_embeddings_file_path" && -f "$legacy_indexed_paths_file_path" && -f "$legacy_skipped_images_file_path" ]]; then
    embeddings_file_path="$legacy_embeddings_file_path"
    indexed_paths_file_path="$legacy_indexed_paths_file_path"
    skipped_images_file_path="$legacy_skipped_images_file_path"
  fi

  log_step "Preparing gallery bank"
  log_kv "gallery_dir=$gallery_dir"
  log_kv "precision=$precision"
  log_kv "model_dir=$MLX_MODEL_DIR"
  if [[ -n "$weights_file_path" && -f "$weights_file_path" ]]; then
    log_kv "weights_file_path=$weights_file_path"
  else
    log_kv "weights_file_path=none"
  fi

  if [[ "${FORCE:-0}" == "1" || ! -f "$embeddings_file_path" || ! -f "$indexed_paths_file_path" || ! -f "$skipped_images_file_path" ]]; then
    log_step "Encoding gallery images"
    encode_cmd=(
      "$PYTHON_BIN_PATH" "$ROOT_DIR/deployment/encode_gallery.py"
      --gallery-path "$gallery_dir"
      --model-path "$MLX_MODEL_DIR"
      --precision "$precision"
      --batch-size "${ENCODE_BATCH_SIZE:-8}"
      --embeddings-output "$embeddings_file_path"
      --paths-output "$indexed_paths_file_path"
      --skipped-output "$skipped_images_file_path"
    )
    if [[ -n "$weights_file_path" && -f "$weights_file_path" ]]; then
      encode_cmd+=(--weights-file "$weights_file_path")
    fi
    "${encode_cmd[@]}"
  else
    log_step "Reusing existing gallery bank"
    log_kv "embeddings_file_path=$embeddings_file_path"
    log_kv "indexed_paths_file_path=$indexed_paths_file_path"
    log_kv "skipped_images_file_path=$skipped_images_file_path"
  fi

  log_step "Writing search config"
  create_cmd=(
    "$PYTHON_BIN_PATH" "$ROOT_DIR/deployment/create_index.py"
    --gallery-path "$gallery_dir"
    --model-path "$MLX_MODEL_DIR"
    --precision "$precision"
    --embeddings-file "$embeddings_file_path"
    --indexed-paths-file "$indexed_paths_file_path"
    --skipped-file "$skipped_images_file_path"
    --config-output "$config_file_path"
  )
  if [[ -n "$weights_file_path" && -f "$weights_file_path" ]]; then
    create_cmd+=(--weights-file "$weights_file_path")
  fi
  if [[ -n "$metadata_manifest_file_path" && -f "$metadata_manifest_file_path" ]]; then
    create_cmd+=(--metadata-manifest "$metadata_manifest_file_path")
  fi
  "${create_cmd[@]}"
}
