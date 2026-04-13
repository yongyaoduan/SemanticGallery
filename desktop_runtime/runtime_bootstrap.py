from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from desktop_runtime.paths import (
    BUNDLED_RESOURCES_DIR_ENV_VAR,
    AppPaths,
    build_app_paths_from_support_dir,
)
from desktop_runtime.progress import ProgressEvent
from desktop_runtime.runtime_setup import RuntimeDownloader, RuntimeSetup


SETUP_PREFIX = "SETUP "


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the SemanticGallery runtime and launch the sidecar.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=36168)
    parser.add_argument("--workspace-root", default=Path(__file__).resolve().parents[1].as_posix())
    parser.add_argument("--index-db", default=None)
    parser.add_argument("--bundled-resources-dir", default=None)
    return parser.parse_args()


def build_runtime_paths(
    workspace_root: Path,
    *,
    bundled_resources_dir: Path | None = None,
) -> AppPaths:
    runtime_root = workspace_root.expanduser().resolve()
    support_root = runtime_root.parent if runtime_root.name == "runtime" else runtime_root
    return build_app_paths_from_support_dir(
        support_root,
        bundled_resources_dir=bundled_resources_dir,
    )


def emit_progress_line(event: ProgressEvent) -> None:
    print(f"{SETUP_PREFIX}{json.dumps(event.to_payload(), ensure_ascii=False, separators=(',', ':'))}", flush=True)


def build_sidecar_command(
    paths: AppPaths,
    *,
    host: str,
    port: int,
    index_db_path: Path | None = None,
) -> list[str]:
    command = [
        (paths.runtime_dir / ".venv" / "bin" / "python").as_posix(),
        "-m",
        "desktop_runtime.sidecar_main",
        "--host",
        host,
        "--port",
        str(port),
        "--workspace-root",
        paths.runtime_dir.as_posix(),
        "--index-db",
        (index_db_path or paths.index_db_path).expanduser().resolve().as_posix(),
    ]
    return command


def main() -> None:
    args = parse_args()
    runtime_root = Path(args.workspace_root).expanduser().resolve()
    bundled_resources_dir = None
    if args.bundled_resources_dir:
        bundled_resources_dir = Path(args.bundled_resources_dir).expanduser().resolve()
        os.environ[BUNDLED_RESOURCES_DIR_ENV_VAR] = bundled_resources_dir.as_posix()
    paths = build_runtime_paths(runtime_root, bundled_resources_dir=bundled_resources_dir)

    setup = RuntimeSetup(paths=paths, downloader=RuntimeDownloader(), emit=emit_progress_line)
    setup.prepare()

    index_db_path = Path(args.index_db).expanduser().resolve() if args.index_db else None
    command = build_sidecar_command(paths, host=args.host, port=args.port, index_db_path=index_db_path)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
