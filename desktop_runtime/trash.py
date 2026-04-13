from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent


def move_paths_to_trash(paths: list[Path]) -> None:
    unique_paths: list[str] = []
    seen_paths: set[str] = set()

    for path in paths:
        resolved = path.expanduser().resolve().as_posix()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_paths.append(resolved)

    if not unique_paths:
        return

    script = dedent(
        """
        ObjC.import("Foundation");

        function moveToTrash(path) {
          const manager = $.NSFileManager.defaultManager;
          const filePath = $(path);
          if (!manager.fileExistsAtPath(filePath)) {
            throw new Error(`Missing path: ${path}`);
          }
          const resultingItem = Ref();
          const error = Ref();
          const ok = manager.trashItemAtURLResultingItemURLError(
            $.NSURL.fileURLWithPath(filePath),
            resultingItem,
            error
          );
          if (!ok) {
            const detail = error[0] ? ObjC.unwrap(error[0].localizedDescription) : "Unknown trash error";
            throw new Error(`${path}: ${detail}`);
          }
        }

        function run(argv) {
          argv.forEach((path) => moveToTrash(path));
        }
        """
    ).strip()

    subprocess.run(
        ["/usr/bin/osascript", "-l", "JavaScript", "-e", script, *unique_paths],
        check=True,
        capture_output=True,
        text=True,
    )
