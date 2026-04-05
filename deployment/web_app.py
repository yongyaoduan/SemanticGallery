from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import mimetypes
import os
import shutil
import sys
from pathlib import Path
from threading import RLock
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps
import uvicorn

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from deployment.search_engine import BaseSearchEngine, build_search_engine
from deployment.search_utils import is_searchable_query

THUMBNAIL_SIZE = (512, 512)
HEIF_SUFFIXES = {".heic", ".heif"}

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional for JPEG/PNG-only galleries
    register_heif_opener = None
else:
    register_heif_opener()


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the local image retrieval UI.")
    parser.add_argument("--config", default="./deployment/search_config.json", help="Search config JSON path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=36168)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


class LocalGalleryServer:
    def __init__(self, search_engine: BaseSearchEngine):
        self.search_engine = search_engine
        self.gallery_path = Path(search_engine.gallery_path).expanduser().resolve()
        self.templates = Jinja2Templates(directory=(Path(__file__).parent / "templates").as_posix())
        self.static_dir = Path(__file__).parent / "static"
        self.static_version = self._build_static_version()
        self.thumbnail_dir = Path(__file__).parent / ".thumb_cache" / self.gallery_path.name
        self.trash_dir = Path(__file__).parent / ".delete_staging" / self.gallery_path.name
        self.metadata_cache: dict[str, dict[str, str]] = {}
        self.engine_lock = RLock()
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def create_app(self) -> FastAPI:
        app = FastAPI(title="SemanticGallery", docs_url=None, redoc_url=None, openapi_url=None)
        app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        @app.get("/")
        async def index(request: Request):
            return self.templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"title": "SemanticGallery", "static_version": self.static_version},
            )

        @app.get("/api/search")
        async def search(query: str = Query("", alias="q"), limit: int = Query(25, ge=1, le=100)):
            text = query.strip()
            if not is_searchable_query(text):
                return {"query": text, "results": []}

            with self.engine_lock:
                results = self.search_engine.search(text, k=limit)
            return {"query": text, "results": self._build_results_payload(results)}

        @app.post("/api/search/image")
        async def search_image(image: UploadFile = File(...), limit: int = Query(25, ge=1, le=100)):
            file_bytes = await image.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Image is empty.")

            query_image = self._read_uploaded_image(file_bytes)
            with self.engine_lock:
                results = self.search_engine.search_by_image(query_image, k=limit)
            return {"query": "", "results": self._build_results_payload(results)}

        @app.get("/api/similar/{image_path:path}")
        async def search_similar(image_path: str, limit: int = Query(25, ge=1, le=100)):
            file_path = self._resolve_gallery_file(image_path)
            with self.engine_lock:
                results = self.search_engine.search_similar(file_path, k=limit)
            return {"query": "", "results": self._build_results_payload(results)}

        @app.get("/api/metadata/{image_path:path}")
        async def metadata(image_path: str):
            file_path = self._resolve_gallery_file(image_path)
            return self._read_image_metadata(file_path)

        @app.get("/images/{image_path:path}")
        async def image(image_path: str):
            file_path = self._resolve_gallery_file(image_path)
            return self._serve_original_image(file_path)

        @app.delete("/api/images/{image_path:path}")
        async def delete_image(image_path: str):
            file_path = self._resolve_gallery_file(image_path)
            with self.engine_lock:
                removed_from_index = self._delete_gallery_file(file_path)
            return {
                "deleted": True,
                "removedFromIndex": removed_from_index,
                "fileName": file_path.name,
                "path": file_path.as_posix(),
            }

        @app.get("/thumbs/{image_path:path}")
        async def thumbnail(image_path: str):
            file_path = self._resolve_gallery_file(image_path)
            thumbnail_path = self._ensure_thumbnail(file_path)
            return FileResponse(thumbnail_path, media_type="image/jpeg")

        return app

    def _build_results_payload(self, results):
        payload = []
        for image_path, _caption in results:
            relative_path = self._relative_image_path(image_path)
            payload.append(
                {
                    "name": Path(image_path).stem,
                    "fileName": Path(image_path).name,
                    "thumbnailUrl": f"/thumbs/{relative_path}",
                    "fullUrl": f"/images/{relative_path}",
                    "metadataUrl": f"/api/metadata/{relative_path}",
                    "deleteUrl": f"/api/images/{relative_path}",
                    "similarUrl": f"/api/similar/{relative_path}",
                }
            )
        return payload

    def _resolve_gallery_file(self, image_path: str) -> Path:
        candidate = (self.gallery_path / image_path).resolve()
        try:
            candidate.relative_to(self.gallery_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Image not found.") from exc

        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Image not found.")
        return candidate

    def _relative_image_path(self, image_path: str) -> str:
        resolved = Path(image_path).expanduser().resolve()
        relative = resolved.relative_to(self.gallery_path).as_posix()
        return quote(relative, safe="/")

    @staticmethod
    def _format_timestamp(raw_value: str) -> str | None:
        text = str(raw_value).strip()
        if not text:
            return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        return text

    def _read_image_metadata(self, file_path: Path) -> dict[str, str]:
        cache_key = file_path.as_posix()
        cached = self.metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        timestamp_label = "文件时间"
        timestamp_value = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        with Image.open(file_path) as image:
            exif = image.getexif()
            for exif_tag in (36867, 36868, 306):
                if exif_tag not in exif:
                    continue
                formatted = self._format_timestamp(exif.get(exif_tag))
                if formatted:
                    timestamp_label = "拍摄时间"
                    timestamp_value = formatted
                    break

        payload = {
            "fileName": file_path.name,
            "path": file_path.as_posix(),
            "timeLabel": timestamp_label,
            "timeValue": timestamp_value,
        }
        self.metadata_cache[cache_key] = payload
        return payload

    @staticmethod
    def _read_uploaded_image(file_bytes: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Unsupported image payload.") from exc

    def _build_static_version(self) -> str:
        targets = [
            self.static_dir / "app.css",
            self.static_dir / "app.js",
            Path(__file__).parent / "templates" / "index.html",
        ]
        latest = max(int(path.stat().st_mtime_ns) for path in targets if path.exists())
        return str(latest)

    def _thumbnail_cache_path(self, file_path: Path) -> Path:
        relative = file_path.relative_to(self.gallery_path).as_posix()
        digest = hashlib.sha1(
            f"{relative}:{file_path.stat().st_mtime_ns}:{THUMBNAIL_SIZE[0]}".encode("utf-8")
        ).hexdigest()
        return self.thumbnail_dir / f"{digest}.jpg"

    def _staging_path(self, file_path: Path) -> Path:
        relative = file_path.relative_to(self.gallery_path)
        staged_name = f"{uuid4().hex}_{relative.name}"
        return (self.trash_dir / relative.parent / staged_name).resolve()

    def _delete_gallery_file(self, file_path: Path) -> bool:
        thumbnail_path = self._thumbnail_cache_path(file_path)
        staged_path = self._staging_path(file_path)
        staged_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(file_path.as_posix(), staged_path.as_posix())
        try:
            removed_from_index = self.search_engine.delete_image(file_path)
            self.metadata_cache.pop(file_path.as_posix(), None)
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            staged_path.unlink(missing_ok=True)
            return removed_from_index
        except Exception:
            if staged_path.exists():
                shutil.move(staged_path.as_posix(), file_path.as_posix())
            raise

    def _ensure_thumbnail(self, file_path: Path) -> Path:
        target = self._thumbnail_cache_path(file_path)
        if target.exists():
            return target

        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            image.save(target, format="JPEG", quality=88, optimize=True)
        return target

    @staticmethod
    def _pil_response(file_path: Path) -> Response:
        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    def _serve_original_image(self, file_path: Path) -> Response:
        if file_path.suffix.lower() in HEIF_SUFFIXES:
            if register_heif_opener is None:
                raise HTTPException(status_code=415, detail="HEIF support is not installed.")
            return self._pil_response(file_path)

        media_type, _ = mimetypes.guess_type(file_path.name)
        return FileResponse(file_path, media_type=media_type or "application/octet-stream")


def build_app(config_path: str, device: str) -> FastAPI:
    search_engine = build_search_engine(config_path=config_path, device=device)
    server = LocalGalleryServer(search_engine)
    return server.create_app()


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting SemanticGallery on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(
        build_app(config_path=args.config, device=args.device),
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
