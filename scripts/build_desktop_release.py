from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_DIR = REPO_ROOT / "desktop"
TAURI_DIR = DESKTOP_DIR / "src-tauri"
RESOURCES_DIR = TAURI_DIR / "resources"
BUILD_BUNDLE_DIR = TAURI_DIR / "target" / "release" / "bundle"


@dataclass
class NotaryCredentials:
    args: list[str]
    temp_files: list[Path]


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    return completed.stdout or ""


def find_signing_identities(security_output: str) -> list[str]:
    identities: list[str] = []
    for line in security_output.splitlines():
        stripped = line.strip()
        if '"' not in stripped:
            continue
        try:
            identities.append(stripped.split('"')[1])
        except IndexError:
            continue
    return identities


def is_developer_id(identity: str) -> bool:
    return identity.startswith("Developer ID Application:")


def select_signing_identity(
    identities: Iterable[str],
    explicit_identity: str | None = None,
    *,
    require_developer_id: bool = False,
) -> str | None:
    available = list(identities)
    if explicit_identity:
        if explicit_identity not in available:
            raise RuntimeError(f"requested signing identity not found: {explicit_identity}")
        if require_developer_id and not is_developer_id(explicit_identity):
            raise RuntimeError("a Developer ID Application identity is required for release signing")
        return explicit_identity

    preferred = [identity for identity in available if is_developer_id(identity)]
    if preferred:
        return preferred[0]

    if require_developer_id:
        raise RuntimeError("no Developer ID Application identity is available")

    return available[0] if available else None


def _write_inline_secret(content: str, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    handle.write(content)
    handle.flush()
    handle.close()
    return Path(handle.name)


def build_notary_credentials(env: dict[str, str]) -> NotaryCredentials | None:
    api_key_path = env.get("APPLE_API_KEY_PATH")
    api_key_inline = env.get("APPLE_API_KEY")
    api_key_id = env.get("APPLE_API_KEY_ID")
    api_issuer = env.get("APPLE_API_ISSUER")
    apple_id = env.get("APPLE_ID")
    apple_password = env.get("APPLE_PASSWORD")
    apple_team_id = env.get("APPLE_TEAM_ID")

    temp_files: list[Path] = []
    if api_key_id and (api_key_path or api_key_inline):
        key_path = Path(api_key_path) if api_key_path else _write_inline_secret(api_key_inline or "", ".p8")
        if not api_key_path:
            temp_files.append(key_path)
        args = ["--key", str(key_path), "--key-id", api_key_id]
        if api_issuer:
            args.extend(["--issuer", api_issuer])
        return NotaryCredentials(args=args, temp_files=temp_files)

    if apple_id and apple_password and apple_team_id:
        return NotaryCredentials(
            args=["--apple-id", apple_id, "--password", apple_password, "--team-id", apple_team_id],
            temp_files=[],
        )

    return None


def stage_uv_binary() -> Path:
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        raise RuntimeError("uv is required on PATH to stage the bundled helper")
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESOURCES_DIR / "uv"
    shutil.copy2(uv_binary, destination)
    destination.chmod(0o755)
    return destination


def clear_previous_outputs(output_dir: Path) -> None:
    for app_dir in (BUILD_BUNDLE_DIR / "macos").glob("*.app"):
        shutil.rmtree(app_dir, ignore_errors=True)
    for dmg_file in (BUILD_BUNDLE_DIR / "dmg").glob("*.dmg"):
        dmg_file.unlink(missing_ok=True)
    for release_file in output_dir.glob("SemanticGallery-macos-arm64*"):
        if release_file.is_dir():
            shutil.rmtree(release_file, ignore_errors=True)
        else:
            release_file.unlink(missing_ok=True)


def signable_paths(app_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in app_path.rglob("*"):
        if path == app_path or not path.exists():
            continue
        if path.suffix in {".app", ".framework", ".xpc"} and path.is_dir():
            candidates.append(path)
            continue
        if not path.is_file():
            continue
        if path.suffix not in {".dylib", ".so"} and not os.access(path, os.X_OK):
            continue
        description = run(["file", "-b", str(path)]).strip()
        if "Mach-O" in description:
            candidates.append(path)
    candidates.sort(key=lambda item: len(item.parts), reverse=True)
    return candidates


def codesign_path(path: Path, identity: str, *, deep: bool = False) -> None:
    command = [
        "codesign",
        "--force",
        "--sign",
        identity,
        "--timestamp",
        "--options",
        "runtime",
    ]
    if deep:
        command.extend(["--deep"])
    command.append(str(path))
    subprocess.run(command, check=True)


def verify_app(app_path: Path) -> None:
    subprocess.run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app_path)], check=True)


def archive_app(app_path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app_path), str(destination)],
        check=True,
    )
    return destination


def notarize(archive_path: Path, credentials: NotaryCredentials) -> None:
    command = ["xcrun", "notarytool", "submit", str(archive_path), "--wait", *credentials.args]
    subprocess.run(command, check=True)


def staple(path: Path) -> None:
    subprocess.run(["xcrun", "stapler", "staple", str(path)], check=True)


def discover_single(pattern: str, *, preferred_name: str | None = None) -> Path:
    matches = sorted(BUILD_BUNDLE_DIR.glob(pattern))
    if not matches:
        raise RuntimeError(f"unable to find build output for pattern: {pattern}")
    if preferred_name:
        for match in matches:
            if match.name == preferred_name:
                return match
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, sign, and optionally notarize the macOS desktop release.")
    parser.add_argument("--allow-unsigned", action="store_true", help="build without signing when no identity is available")
    parser.add_argument("--require-developer-id", action="store_true", help="require a Developer ID Application identity")
    parser.add_argument("--require-notarization", action="store_true", help="fail when notarization credentials are missing")
    parser.add_argument("--skip-notarize", action="store_true", help="skip notarization even if credentials are available")
    parser.add_argument("--output-dir", default="dist/release", help="directory for release artifacts relative to the repo root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_uv_binary()
    clear_previous_outputs(output_dir)
    run([sys.executable, str(DESKTOP_DIR / "scripts" / "generate_icons.py")], cwd=REPO_ROOT)
    run(["npm", "run", "tauri:build", "--", "--bundles", "app,dmg"], cwd=DESKTOP_DIR, capture_output=False)

    app_path = discover_single("macos/*.app", preferred_name="SemanticGallery.app")
    dmg_path = discover_single("dmg/*.dmg")

    identity_output = run(["security", "find-identity", "-v", "-p", "codesigning"])
    identity = select_signing_identity(
        find_signing_identities(identity_output),
        explicit_identity=os.environ.get("APPLE_SIGNING_IDENTITY"),
        require_developer_id=args.require_developer_id,
    )

    if identity is None:
        if not args.allow_unsigned:
            raise RuntimeError("no usable signing identity is available")
    else:
        for nested in signable_paths(app_path):
            codesign_path(nested, identity, deep=nested.is_dir())
        codesign_path(app_path, identity)
        verify_app(app_path)

    release_app_zip = archive_app(app_path, output_dir / "SemanticGallery-macos-arm64.app.zip")
    release_dmg = output_dir / "SemanticGallery-macos-arm64.dmg"
    shutil.copy2(dmg_path, release_dmg)

    credentials = build_notary_credentials(dict(os.environ))
    if args.skip_notarize:
        credentials = None

    if credentials is None:
        if args.require_notarization:
            raise RuntimeError("notarization credentials are missing")
    else:
        try:
            notarize(release_app_zip, credentials)
            staple(app_path)
            notarize(release_dmg, credentials)
            staple(release_dmg)
            release_app_zip = archive_app(app_path, output_dir / "SemanticGallery-macos-arm64.app.zip")
        finally:
            for temp_file in credentials.temp_files:
                temp_file.unlink(missing_ok=True)

    print(release_app_zip)
    print(release_dmg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
