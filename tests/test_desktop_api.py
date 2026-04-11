from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from desktop_runtime.api import build_app
from desktop_runtime.index_store import IndexStore
from desktop_runtime.service import DesktopService
from desktop_runtime.stage2_jobs import Stage2Error


class FakeEncoder:
    def encode_image(self, path: Path) -> np.ndarray:
        if "cat" in path.stem.lower():
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    def encode_text(self, query_text: str) -> np.ndarray:
        if "cat" in query_text.lower():
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


class FakeStage2Job:
    def __init__(self, *, result: str = "stage2-signature", error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[Path] = []

    def run(self, folder_path: Path) -> str:
        self.calls.append(folder_path)
        if self.error is not None:
            raise self.error
        return self.result


class BrokenEncoder(FakeEncoder):
    def encode_image(self, path: Path) -> np.ndarray:
        raise RuntimeError(f"cannot encode {path.name}")


class RecordingEncoderLoader:
    def __init__(self):
        self.calls: list[tuple[Path | None, str]] = []

    def __call__(self, folder_path: Path | None, encoder_signature: str) -> FakeEncoder:
        resolved = folder_path.resolve() if folder_path is not None else None
        self.calls.append((resolved, encoder_signature))
        return FakeEncoder()


class FakeEventService:
    def runtime_status(self) -> dict[str, object]:
        return {
            "setupStatus": "ready",
            "activeFolder": None,
            "activeEncoderSignature": "stage1",
            "lastTaskMessage": "Ready",
            "indexedImageCount": 0,
        }

    async def iter_events(self):
        yield "event: runtime\ndata: {\"message\":\"ready\"}\n\n"


class DesktopApiTests(unittest.TestCase):
    def _build_service(
        self,
        root: Path,
        *,
        stage2_job: FakeStage2Job | None = None,
        encoder_loader: RecordingEncoderLoader | None = None,
    ) -> DesktopService:
        store = IndexStore.connect(root / "index.sqlite3")
        store.migrate()
        return DesktopService(
            store=store,
            encoder=None if encoder_loader is not None else FakeEncoder(),
            encoder_loader=encoder_loader,
            stage2_job=stage2_job or FakeStage2Job(),
        )

    def test_runtime_status_endpoint_returns_setup_state(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            service = self._build_service(root)
            service.setup_status = "ready"
            client = TestClient(build_app(service))

            response = client.get("/api/runtime/status")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["setupStatus"], "ready")
            self.assertEqual(response.json()["activeEncoderSignature"], "stage1")

    def test_select_folder_builds_searchable_view_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            dog = folder / "dog.jpg"
            cat.write_bytes(b"cat")
            dog.write_bytes(b"dog")

            client = TestClient(build_app(self._build_service(root)))

            select_response = client.post("/api/folders/select", json={"folderPath": folder.as_posix()})
            search_response = client.get("/api/search", params={"q": "cat", "limit": 5})
            metadata_response = client.get("/api/metadata/cat.jpg")

            self.assertEqual(select_response.status_code, 200)
            self.assertEqual(select_response.json()["activeFolder"], folder.resolve().as_posix())
            self.assertEqual(search_response.status_code, 200)
            self.assertEqual(search_response.json()["results"][0]["path"], cat.resolve().as_posix())
            self.assertEqual(metadata_response.status_code, 200)
            self.assertEqual(metadata_response.json()["fileName"], "cat.jpg")

    def test_refresh_active_folder_skips_when_lightweight_scan_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            (folder / "cat.jpg").write_bytes(b"cat")
            (folder / "dog.jpg").write_bytes(b"dog")

            service = self._build_service(root)
            service.set_active_folder(folder.as_posix())

            skipped = service.refresh_active_folder(lightweight=True)
            (folder / "bird.jpg").write_bytes(b"bird")
            refreshed = service.refresh_active_folder(lightweight=True)

            self.assertTrue(skipped["skipped"])
            self.assertFalse(refreshed["skipped"])
            self.assertEqual(refreshed["indexedImageCount"], 3)

    def test_select_folder_failure_restores_previous_active_folder(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            good_folder = root / "good-gallery"
            bad_folder = root / "bad-gallery"
            good_folder.mkdir()
            bad_folder.mkdir()
            (good_folder / "cat.jpg").write_bytes(b"cat")
            (bad_folder / "dog.jpg").write_bytes(b"dog")

            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            service = DesktopService(store=store, encoder=BrokenEncoder(), stage2_job=FakeStage2Job())
            service.active_folder = good_folder.resolve()

            client = TestClient(build_app(service))
            response = client.post("/api/folders/select", json={"folderPath": bad_folder.as_posix()})

            self.assertEqual(response.status_code, 400)
            self.assertEqual(service.active_folder, good_folder.resolve())
            self.assertIn("Failed to index the selected folder", response.json()["detail"])

    def test_stage2_endpoint_returns_readable_error(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            (folder / "cat.jpg").write_bytes(b"cat")

            stage2_job = FakeStage2Job(
                error=Stage2Error(
                    "This folder currently has 1 supported images. Stage 2 adaptation is only useful for folders with at least 100 images, so the app will skip adaptation for now."
                )
            )
            service = self._build_service(root, stage2_job=stage2_job)
            service.active_folder = folder.resolve()
            client = TestClient(build_app(service))

            response = client.post("/api/stage2/run")

            self.assertEqual(response.status_code, 400)
            self.assertIn("at least 100 images", response.json()["detail"])

    def test_stage2_endpoint_switches_active_encoder_signature_after_success(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"cat-{index}.jpg").write_bytes(b"cat")

            stage2_job = FakeStage2Job(result="stage2-signature")
            service = self._build_service(root, stage2_job=stage2_job)
            client = TestClient(build_app(service))

            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})
            response = client.post("/api/stage2/run")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["activeEncoderSignature"], "stage2-signature")
            self.assertEqual(stage2_job.calls, [folder.resolve()])

    def test_folder_selection_restores_folder_specific_encoder_signature_after_stage2(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            first_folder = root / "gallery-one"
            second_folder = root / "gallery-two"
            first_folder.mkdir()
            second_folder.mkdir()
            for index in range(100):
                (first_folder / f"cat-{index}.jpg").write_bytes(f"cat-{index}".encode("utf-8"))
            (second_folder / "dog.jpg").write_bytes(b"dog")

            encoder_loader = RecordingEncoderLoader()
            service = self._build_service(
                root,
                stage2_job=FakeStage2Job(result="stage2-gallery-one"),
                encoder_loader=encoder_loader,
            )
            client = TestClient(build_app(service))

            first_select = client.post("/api/folders/select", json={"folderPath": first_folder.as_posix()})
            stage2_response = client.post("/api/stage2/run")
            second_select = client.post("/api/folders/select", json={"folderPath": second_folder.as_posix()})
            restored_select = client.post("/api/folders/select", json={"folderPath": first_folder.as_posix()})

            self.assertEqual(first_select.status_code, 200)
            self.assertEqual(stage2_response.status_code, 200)
            self.assertEqual(second_select.status_code, 200)
            self.assertEqual(second_select.json()["activeEncoderSignature"], "stage1")
            self.assertEqual(restored_select.status_code, 200)
            self.assertEqual(restored_select.json()["activeEncoderSignature"], "stage2-gallery-one")
            self.assertEqual(encoder_loader.calls[-1], (first_folder.resolve(), "stage2-gallery-one"))

    def test_events_endpoint_streams_server_sent_events(self):
        client = TestClient(build_app(FakeEventService()))

        with client.stream("GET", "/api/events") as response:
            lines = list(response.iter_lines())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: runtime", lines)
        self.assertIn('data: {"message":"ready"}', lines)


if __name__ == "__main__":
    unittest.main()
