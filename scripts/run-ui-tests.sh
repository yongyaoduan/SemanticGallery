#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-/tmp/semanticgallery-ui}"
PATCHED_XCTESTRUN_PATH="$DERIVED_DATA_PATH/Build/Products/SemanticGallery_fixed.xctestrun"
UI_TEST_FIXTURE_ROOT="${SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT:-/tmp/semanticgallery-ui-fixtures}"
DEMO_GALLERY_CACHE_ROOT="${SEMANTICGALLERY_DEMO_GALLERY_CACHE_ROOT:-/tmp/semanticgallery-demo-gallery-cache}"
ARTIFACT_FIXTURE_ROOT="${SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT:-/tmp/semanticgallery-ui-artifacts}"
UI_TEST_EVIDENCE_ROOT="${SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT:-/tmp/semanticgallery-validation}"
RESULT_BUNDLE_PATH="${SEMANTICGALLERY_UI_TEST_RESULT_BUNDLE_PATH:-/tmp/semanticgallery-ui-results.xcresult}"
UI_TEST_CLEAN_DERIVED_DATA="${SEMANTICGALLERY_UI_TEST_CLEAN_DERIVED_DATA:-0}"
UI_TEST_TARGET_APP_PATH="${SEMANTICGALLERY_UI_TEST_TARGET_APP_PATH:-__TESTROOT__/Debug/SemanticGallery.app}"
UI_TEST_ONLY_TESTING="${SEMANTICGALLERY_UI_TEST_ONLY_TESTING:-}"

prepare_image_fixture() {
  python "$ROOT_DIR/scripts/prepare_demo_gallery.py" \
    --fixture-root "$UI_TEST_FIXTURE_ROOT" \
    --cache-root "$DEMO_GALLERY_CACHE_ROOT"
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
"$ROOT_DIR/scripts/prepare-bundled-artifacts.sh" "$ARTIFACT_FIXTURE_ROOT"
prepare_image_fixture
prepare_semantic_search_fixture
chmod -R u+rwX,go+rX "$UI_TEST_FIXTURE_ROOT" "$ARTIFACT_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT="$ARTIFACT_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT="$UI_TEST_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT="$UI_TEST_EVIDENCE_ROOT"
export SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT="$ARTIFACT_FIXTURE_ROOT"
export SEMANTICGALLERY_UI_TEST_TARGET_APP_PATH_RESOLVED="$DERIVED_DATA_PATH/Build/Products/Debug/SemanticGallery.app"

XCODEBUILD_TEST_SELECTION_ARGS=()
if [[ -n "$UI_TEST_ONLY_TESTING" ]]; then
  IFS=',' read -r -a only_testing_targets <<< "$UI_TEST_ONLY_TESTING"
  for only_testing_target in "${only_testing_targets[@]}"; do
    [[ -n "$only_testing_target" ]] || continue
    XCODEBUILD_TEST_SELECTION_ARGS+=("-only-testing:$only_testing_target")
  done
fi

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
  "${XCODEBUILD_TEST_SELECTION_ARGS[@]}" \
  -resultBundlePath "$RESULT_BUNDLE_PATH"
