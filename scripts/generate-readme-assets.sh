#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_ROOT="${SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT:-/tmp/semanticgallery-ui-fixtures}"
DEMO_CACHE_ROOT="${SEMANTICGALLERY_DEMO_GALLERY_CACHE_ROOT:-/tmp/semanticgallery-demo-gallery-cache}"
EVIDENCE_ROOT="${SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT:-/tmp/semanticgallery-validation}"
RESULT_BUNDLE_PATH="${SEMANTICGALLERY_UI_TEST_RESULT_BUNDLE_PATH:-/tmp/semanticgallery-readme-assets.xcresult}"
ASSET_ROOT="$ROOT_DIR/docs/assets/readme"
ATTACHMENT_ROOT="$(mktemp -d /tmp/semanticgallery-readme-attachments.XXXXXX)"
FRAME_ROOT="$(mktemp -d /tmp/semanticgallery-readme-frames.XXXXXX)"
trap 'rm -rf "$FRAME_ROOT" "$ATTACHMENT_ROOT"' EXIT

mkdir -p "$ASSET_ROOT"
rm -rf "$EVIDENCE_ROOT" "$RESULT_BUNDLE_PATH"

python "$ROOT_DIR/scripts/prepare_demo_gallery.py" \
  --fixture-root "$FIXTURE_ROOT" \
  --cache-root "$DEMO_CACHE_ROOT"

SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT="$FIXTURE_ROOT" \
SEMANTICGALLERY_DEMO_GALLERY_CACHE_ROOT="$DEMO_CACHE_ROOT" \
SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT="$EVIDENCE_ROOT" \
SEMANTICGALLERY_UI_TEST_RESULT_BUNDLE_PATH="$RESULT_BUNDLE_PATH" \
SEMANTICGALLERY_UI_TEST_ONLY_TESTING="SemanticGalleryUITests/PhaseAFlowTests/testReadmeDemoCapturesKeyScreens" \
"$ROOT_DIR/scripts/run-ui-tests.sh"

xcrun xcresulttool export attachments \
  --test-id "PhaseAFlowTests/testReadmeDemoCapturesKeyScreens()" \
  --path "$RESULT_BUNDLE_PATH" \
  --output-path "$ATTACHMENT_ROOT" >/dev/null

python - "$ATTACHMENT_ROOT" "$ASSET_ROOT" <<'PY'
from pathlib import Path
import json
import shutil
import sys

attachment_root = Path(sys.argv[1])
asset_root = Path(sys.argv[2])
manifest_path = attachment_root / "manifest.json"
manifest = json.loads(manifest_path.read_text())
attachments = manifest[0]["attachments"]

targets = {
    "01-main-interface": "readme-main-interface.png",
    "02-open-settings": "readme-open-settings.png",
    "03-folder-selected": "readme-folder-selected.png",
    "04-indexing": "readme-indexing.png",
    "05-ready-to-search": "readme-ready-to-search.png",
    "06-search-portrait": "readme-search-portrait.png",
    "07-search-winter": "readme-search-winter.png",
    "08-image-search": "readme-image-search.png",
}

for prefix, destination_name in targets.items():
    match = next(
        (
            item
            for item in attachments
            if item["suggestedHumanReadableName"].startswith(prefix)
        ),
        None,
    )
    if match is None:
        raise SystemExit(f"Could not find README screenshot attachment with prefix {prefix!r}.")
    source = attachment_root / match["exportedFileName"]
    destination = asset_root / destination_name
    shutil.copyfile(source, destination)
PY

cp "$FIXTURE_ROOT/SemanticGalleryUITestReadmeDemo/sources.json" "$ASSET_ROOT/demo-gallery-sources.json"
RIGHT_CLICK_SCREENSHOT="$ASSET_ROOT/readme-right-click-open.png"
if [[ ! -f "$RIGHT_CLICK_SCREENSHOT" ]]; then
  echo "Missing README right-click screenshot at $RIGHT_CLICK_SCREENSHOT" >&2
  exit 1
fi

cat > "$FRAME_ROOT/frames.txt" <<EOF
file '$RIGHT_CLICK_SCREENSHOT'
duration 1.6
file '$ASSET_ROOT/readme-main-interface.png'
duration 1.6
file '$ASSET_ROOT/readme-open-settings.png'
duration 1.6
file '$ASSET_ROOT/readme-folder-selected.png'
duration 1.6
file '$ASSET_ROOT/readme-indexing.png'
duration 1.6
file '$ASSET_ROOT/readme-ready-to-search.png'
duration 1.6
file '$ASSET_ROOT/readme-search-portrait.png'
duration 1.6
file '$ASSET_ROOT/readme-search-winter.png'
duration 1.6
file '$ASSET_ROOT/readme-image-search.png'
duration 1.6
file '$ASSET_ROOT/readme-image-search.png'
EOF

ffmpeg -y \
  -f concat \
  -safe 0 \
  -i "$FRAME_ROOT/frames.txt" \
  -vf "fps=10,scale=1200:675:force_original_aspect_ratio=decrease:flags=lanczos,pad=1200:675:(ow-iw)/2:(oh-ih)/2:color=0xF3E8D8,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  "$ASSET_ROOT/semanticgallery-demo.gif"
