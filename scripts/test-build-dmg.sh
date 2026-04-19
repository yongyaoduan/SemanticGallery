#!/usr/bin/env bash

set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/semanticgallery-build-dmg-test.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT

create_temp_repo() {
  local repo_root="$1"
  mkdir -p \
    "$repo_root/scripts" \
    "$repo_root/App/SemanticGallery/Assets.xcassets/AppIcon.appiconset" \
    "$repo_root/App/SemanticGallery.xcodeproj"

  cp "$SOURCE_ROOT/scripts/build-dmg.sh" "$repo_root/scripts/build-dmg.sh"
  chmod +x "$repo_root/scripts/build-dmg.sh"

  local icon_names=(16 32 64 128 256 512 1024)
  local size
  for size in "${icon_names[@]}"; do
    : > "$repo_root/App/SemanticGallery/Assets.xcassets/AppIcon.appiconset/icon-$size.png"
  done
}

create_artifact_root() {
  local artifact_root="$1"
  mkdir -p \
    "$artifact_root/mlx/siglip2-base-patch16-224-f32" \
    "$artifact_root/semanticgallery/stage1" \
    "$artifact_root/semanticgallery/stage2_public_anchor"

  local relative_path
  for relative_path in \
    "mlx/siglip2-base-patch16-224-f32/config.json" \
    "mlx/siglip2-base-patch16-224-f32/tokenizer.json" \
    "mlx/siglip2-base-patch16-224-f32/tokenizer_config.json" \
    "mlx/siglip2-base-patch16-224-f32/special_tokens_map.json" \
    "mlx/siglip2-base-patch16-224-f32/preprocessor_config.json" \
    "semanticgallery/stage1/weights.safetensors" \
    "semanticgallery/stage1/summary.json" \
    "semanticgallery/stage2_public_anchor/semanticgallery-stage2-public-anchor.tar.gz" \
    "semanticgallery/stage2_public_anchor/sample_info.json"; do
    printf 'test fixture\n' > "$artifact_root/$relative_path"
  done
}

create_fake_bin() {
  local bin_root="$1"
  mkdir -p "$bin_root"

  cat <<'EOF' > "$bin_root/xcodebuild"
#!/usr/bin/env bash
set -euo pipefail

derived_data_path=""
while (($#)); do
  if [[ "$1" == "-derivedDataPath" ]]; then
    derived_data_path="$2"
    shift 2
  else
    shift
  fi
done

app_root="$derived_data_path/Build/Products/Release/SemanticGallery.app/Contents/Resources/SemanticGalleryArtifacts"
mkdir -p "$app_root"
cp -R "$SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT"/. "$app_root"
EOF

  cat <<'EOF' > "$bin_root/iconutil"
#!/usr/bin/env bash
set -euo pipefail

output_path=""
while (($#)); do
  if [[ "$1" == "-o" ]]; then
    output_path="$2"
    shift 2
  else
    shift
  fi
done

mkdir -p "$(dirname "$output_path")"
: > "$output_path"
EOF

  cat <<'EOF' > "$bin_root/SetFile"
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF

  cat <<'EOF' > "$bin_root/hdiutil"
#!/usr/bin/env bash
set -euo pipefail

target_path="${!#}"
mkdir -p "$(dirname "$target_path")"
: > "$target_path"
EOF

  chmod +x "$bin_root/xcodebuild" "$bin_root/iconutil" "$bin_root/SetFile" "$bin_root/hdiutil"
}

run_build() {
  local repo_root="$1"
  local artifact_root="$2"
  local fake_bin="$3"
  local derived_root="$4"

  (
    cd "$repo_root"
    PATH="$fake_bin:$PATH" \
    DERIVED_DATA_PATH="$derived_root" \
    SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT="$artifact_root" \
    ./scripts/build-dmg.sh
  )
}

assert_matches() {
  local value="$1"
  local pattern="$2"
  local message="$3"

  if [[ ! "$value" =~ $pattern ]]; then
    echo "$message" >&2
    echo "value: $value" >&2
    echo "pattern: $pattern" >&2
    exit 1
  fi
}

assert_equals() {
  local expected="$1"
  local actual="$2"
  local message="$3"

  if [[ "$expected" != "$actual" ]]; then
    echo "$message" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
}

main() {
  local default_repo="$TEST_ROOT/default"
  local default_artifacts="$TEST_ROOT/default-artifacts"
  local default_bin="$TEST_ROOT/default-bin"

  create_temp_repo "$default_repo"
  create_artifact_root "$default_artifacts"
  create_fake_bin "$default_bin"

  local default_output
  default_output="$(run_build "$default_repo" "$default_artifacts" "$default_bin" "$TEST_ROOT/default-derived")"
  assert_matches \
    "$default_output" \
    '.*/dist/SemanticGallery-[0-9]{8}-[0-9]{6}\.dmg$' \
    "Default builds should keep the timestamped dmg name."

  local release_repo="$TEST_ROOT/release"
  local release_artifacts="$TEST_ROOT/release-artifacts"
  local release_bin="$TEST_ROOT/release-bin"

  create_temp_repo "$release_repo"
  create_artifact_root "$release_artifacts"
  create_fake_bin "$release_bin"

  local release_output
  release_output="$(
    cd "$release_repo"
    PATH="$release_bin:$PATH" \
    DERIVED_DATA_PATH="$TEST_ROOT/release-derived" \
    SEMANTICGALLERY_BUNDLED_ARTIFACT_SOURCE_ROOT="$release_artifacts" \
    SEMANTICGALLERY_RELEASE_NAME="SemanticGallery-1.0.0" \
    ./scripts/build-dmg.sh
  )"
  assert_equals \
    "$release_repo/dist/SemanticGallery-1.0.0.dmg" \
    "$release_output" \
    "Release builds should use the configured formal dmg name."

  local latest_target
  latest_target="$(readlink "$release_repo/dist/SemanticGallery-latest.dmg")"
  assert_equals \
    "SemanticGallery-1.0.0.dmg" \
    "$latest_target" \
    "The latest symlink should point to the configured release dmg."
}

main "$@"
