#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PATH="$ROOT_DIR/App/SemanticGallery.xcodeproj"
SCHEME_NAME="SemanticGallery"
APP_NAME="SemanticGallery"
VOLNAME="SemanticGallery"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-/tmp/semanticgallery-release}"
DIST_DIR="$ROOT_DIR/dist"
BUILD_APP_PATH="$DERIVED_DATA_PATH/Build/Products/Release/$APP_NAME.app"
APPICONSET_PATH="$ROOT_DIR/App/SemanticGallery/Assets.xcassets/AppIcon.appiconset"
ARTIFACT_SOURCE_ROOT="${SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT:-$ROOT_DIR/.cache}"
APP_ARTIFACTS_PATH="$BUILD_APP_PATH/Contents/Resources/SemanticGalleryArtifacts"
STAMP="$(date +%Y%m%d-%H%M%S)"
DMG_PATH="$DIST_DIR/$APP_NAME-$STAMP.dmg"
LATEST_PATH="$DIST_DIR/$APP_NAME-latest.dmg"

create_volume_icon() {
  local iconset_root="$1"
  local icns_path="$2"

  mkdir -p "$iconset_root"
  cp "$APPICONSET_PATH/icon-16.png" "$iconset_root/icon_16x16.png"
  cp "$APPICONSET_PATH/icon-32.png" "$iconset_root/icon_16x16@2x.png"
  cp "$APPICONSET_PATH/icon-32.png" "$iconset_root/icon_32x32.png"
  cp "$APPICONSET_PATH/icon-64.png" "$iconset_root/icon_32x32@2x.png"
  cp "$APPICONSET_PATH/icon-128.png" "$iconset_root/icon_128x128.png"
  cp "$APPICONSET_PATH/icon-256.png" "$iconset_root/icon_128x128@2x.png"
  cp "$APPICONSET_PATH/icon-256.png" "$iconset_root/icon_256x256.png"
  cp "$APPICONSET_PATH/icon-512.png" "$iconset_root/icon_256x256@2x.png"
  cp "$APPICONSET_PATH/icon-512.png" "$iconset_root/icon_512x512.png"
  cp "$APPICONSET_PATH/icon-1024.png" "$iconset_root/icon_512x512@2x.png"
  iconutil -c icns "$iconset_root" -o "$icns_path"
}

verify_artifact_source() {
  local root="$1"
  local required_paths=(
    "mlx/siglip2-base-patch16-224-f32/config.json"
    "mlx/siglip2-base-patch16-224-f32/tokenizer.json"
    "mlx/siglip2-base-patch16-224-f32/tokenizer_config.json"
    "mlx/siglip2-base-patch16-224-f32/special_tokens_map.json"
    "mlx/siglip2-base-patch16-224-f32/preprocessor_config.json"
    "semanticgallery/stage1/weights.safetensors"
    "semanticgallery/stage1/summary.json"
    "semanticgallery/stage2_public_anchor/semanticgallery-stage2-public-anchor.tar.gz"
    "semanticgallery/stage2_public_anchor/sample_info.json"
  )

  for relative_path in "${required_paths[@]}"; do
    if [[ ! -s "$root/$relative_path" ]]; then
      echo "Bundled artifact source is missing $relative_path at $root" >&2
      exit 1
    fi
  done
}

mkdir -p "$DIST_DIR"
verify_artifact_source "$ARTIFACT_SOURCE_ROOT"

SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT="$ARTIFACT_SOURCE_ROOT" \
xcodebuild \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME_NAME" \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  build

if [[ ! -d "$BUILD_APP_PATH" ]]; then
  echo "Release app was not built at $BUILD_APP_PATH" >&2
  exit 1
fi

verify_artifact_source "$APP_ARTIFACTS_PATH"

WORK_ROOT="$(mktemp -d /tmp/semanticgallery-dmg.XXXXXX)"
trap 'rm -rf "$WORK_ROOT"' EXIT

STAGING_ROOT="$WORK_ROOT/$VOLNAME"
ICONSET_ROOT="$WORK_ROOT/VolumeIcon.iconset"
ICNS_PATH="$WORK_ROOT/.VolumeIcon.icns"

mkdir -p "$STAGING_ROOT"
cp -R "$BUILD_APP_PATH" "$STAGING_ROOT/$APP_NAME.app"
ln -s /Applications "$STAGING_ROOT/Applications"
create_volume_icon "$ICONSET_ROOT" "$ICNS_PATH"
cp "$ICNS_PATH" "$STAGING_ROOT/.VolumeIcon.icns"
SetFile -a C "$STAGING_ROOT"

hdiutil create \
  -volname "$VOLNAME" \
  -srcfolder "$STAGING_ROOT" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

rm -f "$LATEST_PATH"
ln -s "$(basename "$DMG_PATH")" "$LATEST_PATH"

echo "$DMG_PATH"
