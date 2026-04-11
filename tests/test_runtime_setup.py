from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from desktop_runtime.paths import build_app_paths
from desktop_runtime.runtime_setup import RuntimeSetup, SetupStatus


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def ensure_python_runtime(self, *args, **kwargs):
        self.calls.append("python")

    def ensure_dependencies(self, *args, **kwargs):
        self.calls.append("deps")

    def ensure_base_model(self, *args, **kwargs):
        self.calls.append("model")

    def ensure_public_anchor(self, *args, **kwargs):
        self.calls.append("anchor")


class RuntimeSetupTests(unittest.TestCase):
    def test_prepare_runtime_reports_expected_steps_in_order(self):
        events = []
        downloader = FakeDownloader()
        with tempfile.TemporaryDirectory(prefix="sg-setup-") as tmp_dir:
            paths = build_app_paths(Path(tmp_dir))
            setup = RuntimeSetup(paths=paths, downloader=downloader, emit=events.append)

            status = setup.prepare()

        self.assertEqual(status, SetupStatus.READY)
        self.assertEqual(downloader.calls, ["python", "deps", "model", "anchor"])
        self.assertEqual(
            [(event.task, event.phase) for event in events if event.phase == "start"],
            [
                ("check-runtime", "start"),
                ("prepare-dependencies", "start"),
                ("prepare-base-model", "start"),
                ("prepare-public-anchor", "start"),
                ("finish-setup", "start"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
