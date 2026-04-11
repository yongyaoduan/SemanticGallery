from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BUNDLED_RESOURCES_DIR_ENV_VAR = "SEMANTICGALLERY_BUNDLED_RESOURCES_DIR"


@dataclass(frozen=True)
class AppPaths:
    app_name: str
    support_dir: Path
    runtime_dir: Path
    cache_dir: Path
    logs_dir: Path
    stage2_dir: Path
    thumbnails_dir: Path
    index_db_path: Path
    config_dir: Path
    bundled_resources_dir: Path


def resolve_bundled_resources_dir() -> Path:
    override = os.environ.get(BUNDLED_RESOURCES_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "resources"


def build_app_paths(home_dir: Path, app_name: str = "SemanticGallery") -> AppPaths:
    support_dir = home_dir.expanduser().resolve() / "Library" / "Application Support" / app_name
    return AppPaths(
        app_name=app_name,
        support_dir=support_dir,
        runtime_dir=support_dir / "runtime",
        cache_dir=support_dir / "cache",
        logs_dir=support_dir / "logs",
        stage2_dir=support_dir / "stage2",
        thumbnails_dir=support_dir / "thumbs",
        index_db_path=support_dir / "index.sqlite3",
        config_dir=support_dir / "config",
        bundled_resources_dir=resolve_bundled_resources_dir(),
    )


def resolve_uv_binary(paths: AppPaths, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return paths.bundled_resources_dir / "uv"
