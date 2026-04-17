#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-/tmp/semanticgallery-ui}"
PATCHED_XCTESTRUN_PATH="$DERIVED_DATA_PATH/Build/Products/SemanticGallery_fixed.xctestrun"
UI_TEST_FIXTURE_ROOT="${SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT:-/tmp/semanticgallery-ui-fixtures}"
IMAGE_SOURCE_ROOT="${SEMANTICGALLERY_UI_TEST_IMAGE_SOURCE_ROOT:-/Users/$USER/PythonProjects/phone_pictures}"
ARTIFACT_FIXTURE_ROOT="${SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT:-/tmp/semanticgallery-ui-artifacts}"
UI_TEST_EVIDENCE_ROOT="${SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT:-/tmp/semanticgallery-validation}"
RESULT_BUNDLE_PATH="${SEMANTICGALLERY_UI_TEST_RESULT_BUNDLE_PATH:-/tmp/semanticgallery-ui-results.xcresult}"
UI_TEST_CLEAN_DERIVED_DATA="${SEMANTICGALLERY_UI_TEST_CLEAN_DERIVED_DATA:-0}"
UI_TEST_TARGET_APP_PATH="${SEMANTICGALLERY_UI_TEST_TARGET_APP_PATH:-__TESTROOT__/Debug/SemanticGallery.app}"

download_artifact_file() {
  local repo_type="$1"
  local repo_id="$2"
  local relative_path="$3"
  local filename="$4"
  local destination_dir="$ARTIFACT_FIXTURE_ROOT/$relative_path"
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

prepare_artifact_fixture() {
  mkdir -p "$ARTIFACT_FIXTURE_ROOT"

  download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 config.json
  download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 tokenizer.json
  download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 tokenizer_config.json
  download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 special_tokens_map.json
  download_artifact_file model google/siglip2-base-patch16-224 mlx/siglip2-base-patch16-224-f32 preprocessor_config.json
  download_artifact_file model Lucas20250626/semanticgallery-mlx-siglip2-stage1 semanticgallery/stage1 weights.safetensors
  download_artifact_file model Lucas20250626/semanticgallery-mlx-siglip2-stage1 semanticgallery/stage1 summary.json
  download_artifact_file dataset Lucas20250626/semanticgallery-stage2-public-anchor semanticgallery/stage2_public_anchor semanticgallery-stage2-public-anchor.tar.gz
  download_artifact_file dataset Lucas20250626/semanticgallery-stage2-public-anchor semanticgallery/stage2_public_anchor sample_info.json
}

prepare_image_fixture() {
  python - "$IMAGE_SOURCE_ROOT" "$UI_TEST_FIXTURE_ROOT" <<'PY'
from pathlib import Path
import random
import shutil
import sys

source_root = Path(sys.argv[1]).expanduser().resolve()
fixture_root = Path(sys.argv[2]).expanduser().resolve()
allowed = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
files = [path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
album_fixtures = [
    "SemanticGalleryUITestAlbumPreparation",
    "SemanticGalleryUITestAlbumUsageSearch",
    "SemanticGalleryUITestAlbumDelete",
    "SemanticGalleryUITestAlbumImageSearch",
    "SemanticGalleryUITestAlbumPrivateAdaptationMinimum",
    "SemanticGalleryUITestAlbumWorkspaceValidation",
]
private_fixtures = [
    "SemanticGalleryUITestPrivateAlbumUninstall",
    "SemanticGalleryUITestPrivateAlbumTraining",
]
album_count = 8 * len(album_fixtures)
private_count = 120 * len(private_fixtures)
required_count = album_count + private_count
if len(files) < required_count:
    raise SystemExit(f"Need at least {required_count} source images in {source_root}")

rng = random.Random(42)
selected = rng.sample(files, required_count)
if fixture_root.exists():
    shutil.rmtree(fixture_root)
fixture_root.mkdir(parents=True, exist_ok=True)

cursor = 0
for fixture_name in album_fixtures:
    destination_root = fixture_root / fixture_name
    destination_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 9):
        source = selected[cursor]
        cursor += 1
        shutil.copy2(source, destination_root / f"sample-{index:02d}{source.suffix.lower()}")

for fixture_name in private_fixtures:
    destination_root = fixture_root / fixture_name
    destination_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 121):
        source = selected[cursor]
        cursor += 1
        shutil.copy2(source, destination_root / f"private-{index:03d}{source.suffix.lower()}")
PY
}

prepare_semantic_search_fixture() {
  python - "$UI_TEST_FIXTURE_ROOT" <<'PY'
from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import sys

fixture_root = Path(sys.argv[1]).expanduser().resolve()
destination_root = fixture_root / "SemanticGalleryUITestSemanticYellowCat"
sources = [
    ("study-01.jpg", "https://images.pexels.com/photos/9415244/pexels-photo-9415244.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
    ("study-02.jpg", "https://images.pexels.com/photos/14440674/pexels-photo-14440674.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
    ("study-03.jpg", "https://images.pexels.com/photos/208954/pexels-photo-208954.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
    ("study-04.jpg", "https://images.pexels.com/photos/7543135/pexels-photo-7543135.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
    ("study-05.jpg", "https://images.pexels.com/photos/11774609/pexels-photo-11774609.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
    ("study-06.jpg", "https://images.pexels.com/photos/17078821/pexels-photo-17078821.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
    ("study-07.jpg", "https://images.pexels.com/photos/14701951/pexels-photo-14701951.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"),
]
expected = {filename for filename, _ in sources}
existing = {path.name for path in destination_root.glob("*")} if destination_root.exists() else set()
if existing != expected:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    for filename, url in sources:
        request = Request(url, headers={"User-Agent": "SemanticGalleryUITests"})
        with urlopen(request, timeout=120) as response:
            data = response.read()
        (destination_root / filename).write_bytes(data)
PY
}

cd "$ROOT_DIR"
if [[ "$UI_TEST_CLEAN_DERIVED_DATA" == "1" ]]; then
  rm -rf "$DERIVED_DATA_PATH"
fi
rm -rf "$UI_TEST_EVIDENCE_ROOT" "$RESULT_BUNDLE_PATH"
mkdir -p "$UI_TEST_EVIDENCE_ROOT"
prepare_artifact_fixture
prepare_image_fixture
prepare_semantic_search_fixture
chmod -R u+rwX,go+rX "$UI_TEST_FIXTURE_ROOT" "$ARTIFACT_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT="$ARTIFACT_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT="$UI_TEST_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT="$UI_TEST_EVIDENCE_ROOT"

xcodebuild build-for-testing \
  -project App/SemanticGallery.xcodeproj \
  -scheme SemanticGallery \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$DERIVED_DATA_PATH"

XCTESTRUN_PATH="$(find "$DERIVED_DATA_PATH/Build/Products" -maxdepth 1 -name 'SemanticGallery_*.xctestrun' ! -name 'SemanticGallery_fixed.xctestrun' -print -quit)"
if [[ -z "$XCTESTRUN_PATH" ]]; then
  echo "Could not locate the generated xctestrun file." >&2
  exit 1
fi

rm -f "$PATCHED_XCTESTRUN_PATH"
cp "$XCTESTRUN_PATH" "$PATCHED_XCTESTRUN_PATH"
/usr/libexec/PlistBuddy \
  -c "Set :SemanticGalleryUITests:UITargetAppPath $UI_TEST_TARGET_APP_PATH" \
  "$PATCHED_XCTESTRUN_PATH"

xcodebuild test-without-building \
  -xctestrun "$PATCHED_XCTESTRUN_PATH" \
  -destination 'platform=macOS,arch=arm64' \
  -resultBundlePath "$RESULT_BUNDLE_PATH"
