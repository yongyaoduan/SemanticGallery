#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


ALBUM_FIXTURES = [
    "SemanticGalleryUITestAlbumPreparation",
    "SemanticGalleryUITestAlbumUsageSearch",
    "SemanticGalleryUITestAlbumDelete",
    "SemanticGalleryUITestAlbumImageSearch",
    "SemanticGalleryUITestAlbumPrivateAdaptationMinimum",
    "SemanticGalleryUITestAlbumWorkspaceValidation",
]

PRIVATE_FIXTURES = [
    "SemanticGalleryUITestPrivateAlbumUninstall",
    "SemanticGalleryUITestPrivateAlbumTraining",
]

README_FIXTURE = "SemanticGalleryUITestReadmeDemo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("demo_gallery_manifest.json")),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if len(entries) != 120:
        raise SystemExit(f"Expected 120 demo images in {path}, found {len(entries)}")
    return entries


def interleaved(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    categories = ["people", "landscape", "architecture", "other"]
    buckets = {
        category: [entry for entry in entries if entry["category"] == category]
        for category in categories
    }

    ordered: list[dict[str, str]] = []
    index = 0
    while True:
        appended = False
        for category in categories:
            bucket = buckets[category]
            if index < len(bucket):
                ordered.append(bucket[index])
                appended = True
        if appended is False:
            break
        index += 1
    return ordered


def download_demo_sources(entries: list[dict[str, str]], cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        destination = cache_root / entry["filename"]
        if destination.exists() and destination.stat().st_size > 0:
            continue
        request = Request(entry["image_url"], headers={"User-Agent": "SemanticGalleryDemoAssets"})
        with urlopen(request, timeout=180) as response:
            destination.write_bytes(response.read())


def copy_named_fixture(
    source_entries: list[dict[str, str]],
    cache_root: Path,
    destination_root: Path,
    prefix: str,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(source_entries, start=1):
        source = cache_root / entry["filename"]
        destination = destination_root / f"{prefix}-{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, destination)


def copy_private_fixture(
    entries: list[dict[str, str]],
    cache_root: Path,
    destination_root: Path,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries, start=1):
        source = cache_root / entry["filename"]
        destination = destination_root / f"private-{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)


def copy_readme_fixture(
    entries: list[dict[str, str]],
    cache_root: Path,
    destination_root: Path,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.copy2(cache_root / entry["filename"], destination_root / entry["filename"])


def write_source_manifest(entries: list[dict[str, str]], destination_root: Path) -> None:
    destination_root.joinpath("sources.json").write_text(
        json.dumps(entries, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    fixture_root = Path(args.fixture_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    entries = load_manifest(manifest_path)
    mixed_entries = interleaved(entries)
    download_demo_sources(entries, cache_root)

    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    fixture_root.mkdir(parents=True, exist_ok=True)

    for fixture_index, fixture_name in enumerate(ALBUM_FIXTURES):
        start = fixture_index * 8
        copy_named_fixture(
            mixed_entries[start:start + 8],
            cache_root,
            fixture_root / fixture_name,
            "sample",
        )

    for fixture_name in PRIVATE_FIXTURES:
        copy_private_fixture(entries, cache_root, fixture_root / fixture_name)

    readme_root = fixture_root / README_FIXTURE
    copy_readme_fixture(entries, cache_root, readme_root)
    write_source_manifest(entries, readme_root)


if __name__ == "__main__":
    main()
