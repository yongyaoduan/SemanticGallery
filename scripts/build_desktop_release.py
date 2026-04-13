from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_DIR = REPO_ROOT / "desktop"
TAURI_DIR = DESKTOP_DIR / "src-tauri"
RESOURCES_DIR = TAURI_DIR / "resources"
BUILD_BUNDLE_DIR = TAURI_DIR / "target" / "release" / "bundle"
PRODUCT_NAME = "SemanticGallery"
APP_BUNDLE_NAME = f"{PRODUCT_NAME}.app"
UNINSTALLER_APP_NAME = f"Uninstall {PRODUCT_NAME}.app"
APP_BUNDLE_IDENTIFIER = "com.semanticgallery.desktop"
UNINSTALLER_BUNDLE_IDENTIFIER = "com.semanticgallery.desktop.uninstall"
ICON_FILENAMES = [
    "32x32.png",
    "128x128.png",
    "128x128@2x.png",
    "icon.icns",
    "icon.png",
]


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


def render_uninstaller_applescript(
    app_name: str = PRODUCT_NAME,
    bundle_identifier: str = APP_BUNDLE_IDENTIFIER,
) -> str:
    return (
        dedent(
            f"""
            ObjC.import("Foundation");

            function parentPath(path) {{
              return ObjC.unwrap($(path).stringByDeletingLastPathComponent);
            }}

            function fileExists(path) {{
              return $.NSFileManager.defaultManager.fileExistsAtPath($(path));
            }}

            function uniquePaths(paths) {{
              return [...new Set(paths.filter(Boolean))];
            }}

            function moveToTrash(path) {{
              const manager = $.NSFileManager.defaultManager;
              const resultingItem = Ref();
              const error = Ref();
              manager.trashItemAtURLResultingItemURLError($.NSURL.fileURLWithPath($(path)), resultingItem, error);
            }}

            function run() {{
              const appName = "{app_name}";
              const bundleIdentifier = "{bundle_identifier}";
              const app = Application.currentApplication();
              app.includeStandardAdditions = true;
              app.activate();

              const helperPath = ObjC.unwrap($.NSBundle.mainBundle.bundlePath);
              const helperDir = parentPath(helperPath);
              const helperParent = parentPath(helperDir);
              const homeDir = ObjC.unwrap($.NSHomeDirectory());

              try {{
                app.displayDialog(
                  `Uninstall ${{appName}} and remove its downloaded runtime, models, local indexes, caches, and setup data from this Mac?`,
                  {{
                    withTitle: appName,
                    buttons: ["Cancel", "Uninstall"],
                    defaultButton: "Uninstall",
                    cancelButton: "Cancel"
                  }}
                );
              }} catch (error) {{
                return;
              }}

              try {{
                Application(bundleIdentifier).quit();
              }} catch (error) {{
              }}
              try {{
                app.doShellScript(
                  "/usr/bin/pkill -f 'semanticgallery_desktop' || true; " +
                  "/usr/bin/pkill -f 'desktop_runtime.sidecar_main' || true; " +
                  "/usr/bin/pkill -f 'desktop_runtime.runtime_bootstrap' || true"
                );
              }} catch (error) {{
              }}
              delay(1);

              const appCandidates = [
                `${{helperDir}}/{APP_BUNDLE_NAME}`,
                `${{helperParent}}/{APP_BUNDLE_NAME}`,
                `/Applications/{APP_BUNDLE_NAME}`,
                `${{homeDir}}/Applications/{APP_BUNDLE_NAME}`
              ];
              const dataCandidates = [
                `${{homeDir}}/Library/Application Support/{APP_BUNDLE_IDENTIFIER}`,
                `${{homeDir}}/Library/Application Support/{PRODUCT_NAME}`,
                `${{homeDir}}/Library/Caches/{APP_BUNDLE_IDENTIFIER}`,
                `${{homeDir}}/Library/WebKit/{APP_BUNDLE_IDENTIFIER}`,
                `${{homeDir}}/Library/HTTPStorages/{APP_BUNDLE_IDENTIFIER}`,
                `${{homeDir}}/Library/HTTPStorages/{APP_BUNDLE_IDENTIFIER}.binarycookies`,
                `${{homeDir}}/Library/Preferences/{APP_BUNDLE_IDENTIFIER}.plist`,
                `${{homeDir}}/Library/Saved Application State/{APP_BUNDLE_IDENTIFIER}.savedState`
              ];

              const removalTargets = uniquePaths(
                [...appCandidates, ...dataCandidates].filter((path) => path !== helperPath && fileExists(path))
              );
              if (!removalTargets.length) {{
                app.displayDialog(`${{appName}} did not find an installed app or local runtime data to remove.`, {{
                  withTitle: appName,
                  buttons: ["OK"],
                  defaultButton: "OK"
                }});
                return;
              }}

              removalTargets.forEach((path) => {{
                try {{
                  moveToTrash(path);
                }} catch (error) {{
                }}
              }});

              app.displayDialog(`${{appName}} moved the app and its local runtime data to the Trash.`, {{
                withTitle: appName,
                buttons: ["OK"],
                defaultButton: "OK"
              }});
            }}
            """
        ).strip()
        + "\n"
    )


def stage_uninstaller_app() -> Path:
    helper_root = RESOURCES_DIR / "uninstall"
    helper_root.mkdir(parents=True, exist_ok=True)
    destination = helper_root / UNINSTALLER_APP_NAME
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(render_uninstaller_applescript())
        script_path = Path(handle.name)

    try:
        subprocess.run(["osacompile", "-l", "JavaScript", "-o", str(destination), str(script_path)], check=True)
    finally:
        script_path.unlink(missing_ok=True)

    icon_source = TAURI_DIR / "icons" / "icon.icns"
    applet_icon = destination / "Contents" / "Resources" / "applet.icns"
    if icon_source.is_file():
        shutil.copy2(icon_source, applet_icon)

    info_plist_path = destination / "Contents" / "Info.plist"
    if info_plist_path.is_file():
        with info_plist_path.open("rb") as handle:
            info = plistlib.load(handle)
        info["CFBundleDisplayName"] = f"Uninstall {PRODUCT_NAME}"
        info["CFBundleName"] = f"Uninstall {PRODUCT_NAME}"
        info["CFBundleIdentifier"] = UNINSTALLER_BUNDLE_IDENTIFIER
        with info_plist_path.open("wb") as handle:
            plistlib.dump(info, handle)

    return destination


def icons_are_ready() -> bool:
    icon_dir = TAURI_DIR / "icons"
    return all((icon_dir / name).is_file() for name in ICON_FILENAMES)


def ensure_icons() -> None:
    try:
        run([sys.executable, str(DESKTOP_DIR / "scripts" / "generate_icons.py")], cwd=REPO_ROOT)
    except subprocess.CalledProcessError:
        if not icons_are_ready():
            raise


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


def prepare_release_stage(app_path: Path, stage_root: Path, *, uninstall_helper_path: Path | None = None) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    volume_root = stage_root / PRODUCT_NAME
    if volume_root.exists():
        shutil.rmtree(volume_root, ignore_errors=True)
    volume_root.mkdir()
    shutil.copytree(app_path, volume_root / app_path.name, symlinks=True)
    if uninstall_helper_path and uninstall_helper_path.exists():
        shutil.copytree(uninstall_helper_path, volume_root / uninstall_helper_path.name, symlinks=True)
    applications_link = volume_root / "Applications"
    if not applications_link.exists():
        applications_link.symlink_to("/Applications")
    return volume_root


def build_release_dmg(app_path: Path, destination: Path, *, uninstall_helper_path: Path | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    with tempfile.TemporaryDirectory(prefix="semanticgallery-dmg-stage-") as temp_dir:
        stage_root = Path(temp_dir)
        volume_root = prepare_release_stage(app_path, stage_root, uninstall_helper_path=uninstall_helper_path)
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                PRODUCT_NAME,
                "-srcfolder",
                str(volume_root),
                "-fs",
                "HFS+",
                "-format",
                "UDZO",
                str(destination),
            ],
            check=True,
        )
    return destination


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
    uninstall_helper = stage_uninstaller_app()
    clear_previous_outputs(output_dir)
    ensure_icons()
    run(["npm", "run", "tauri:build", "--", "--bundles", "app"], cwd=DESKTOP_DIR, capture_output=False)

    app_path = discover_single("macos/*.app", preferred_name=APP_BUNDLE_NAME)
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
        codesign_path(uninstall_helper, identity, deep=True)
        verify_app(uninstall_helper)

    release_app_zip = archive_app(app_path, output_dir / "SemanticGallery-macos-arm64.app.zip")

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
            release_app_zip = archive_app(app_path, output_dir / "SemanticGallery-macos-arm64.app.zip")
        finally:
            for temp_file in credentials.temp_files:
                temp_file.unlink(missing_ok=True)

    release_dmg = build_release_dmg(
        app_path,
        output_dir / "SemanticGallery-macos-arm64.dmg",
        uninstall_helper_path=uninstall_helper,
    )
    if credentials is not None:
        notarize(release_dmg, credentials)
        staple(release_dmg)

    print(release_app_zip)
    print(release_dmg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
