from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from desktop_runtime.service import DesktopService, DesktopServiceError
from desktop_runtime.stage2_jobs import Stage2Error


class FolderSelectionPayload(BaseModel):
    folderPath: str


class FolderActivationPayload(BaseModel):
    folderPath: str
    encoderSignature: str = "stage1"


class EncodeTextPayload(BaseModel):
    folderPath: str
    encoderSignature: str = "stage1"
    queryText: str


class EncodeImagePathPayload(BaseModel):
    folderPath: str
    encoderSignature: str = "stage1"
    imagePath: str


class BatchDeletePayload(BaseModel):
    paths: list[str]


def build_app(
    service: DesktopService | object | None = None,
    *,
    on_startup=None,
    on_shutdown=None,
) -> FastAPI:
    service = service or DesktopService()

    def bind_service_loop() -> None:
        attach_event_loop = getattr(service, "attach_event_loop", None)
        if callable(attach_event_loop):
            attach_event_loop(asyncio.get_running_loop())

    async def run_hook(hook) -> None:
        if hook is None:
            return
        result = hook()
        if inspect.isawaitable(result):
            await result

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bind_service_loop()
        await run_hook(on_startup)
        try:
            yield
        finally:
            await run_hook(on_shutdown)

    app = FastAPI(
        title="SemanticGallery Desktop Sidecar",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/runtime/status")
    async def runtime_status():
        bind_service_loop()
        return service.runtime_status()

    @app.post("/api/folders/select")
    async def select_folder(payload: FolderSelectionPayload):
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.set_active_folder, payload.folderPath)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/folders/activate")
    async def activate_folder(payload: FolderActivationPayload):
        try:
            bind_service_loop()
            return await asyncio.to_thread(
                service.activate_folder,
                payload.folderPath,
                payload.encoderSignature,
            )
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/folders/refresh")
    async def refresh_folder():
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.refresh_active_folder, False)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage2/run")
    async def run_stage2():
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.run_stage2_for_active_folder)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/search")
    async def search(query: str = Query("", alias="q"), limit: int = Query(25, ge=1, le=100)):
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.search_text, query, limit)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/search/image")
    async def search_image(image: UploadFile = File(...), limit: int = Query(25, ge=1, le=100)):
        try:
            bind_service_loop()
            file_bytes = await image.read()
            return await asyncio.to_thread(service.search_image, file_bytes, limit, image.filename)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/encode/text")
    async def encode_text(payload: EncodeTextPayload):
        try:
            bind_service_loop()
            vector = await asyncio.to_thread(
                service.encode_text_vector,
                payload.folderPath,
                payload.encoderSignature,
                payload.queryText,
            )
            return {"vector": vector}
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/encode/image-path")
    async def encode_image_path(payload: EncodeImagePathPayload):
        try:
            bind_service_loop()
            vector = await asyncio.to_thread(
                service.encode_image_path_vector,
                payload.folderPath,
                payload.encoderSignature,
                payload.imagePath,
            )
            return {"vector": vector}
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/metadata/{image_path:path}")
    async def metadata(image_path: str):
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.metadata, image_path)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/similar/{image_path:path}")
    async def similar(image_path: str, limit: int = Query(25, ge=1, le=100)):
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.search_similar, image_path, limit)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/thumbs/{image_path:path}")
    async def thumbnail(image_path: str):
        try:
            bind_service_loop()
            thumbnail_path = await asyncio.to_thread(service.thumbnail_path, image_path)
            return FileResponse(thumbnail_path, media_type="image/jpeg")
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/images/{image_path:path}")
    async def image(image_path: str):
        try:
            bind_service_loop()
            asset = await asyncio.to_thread(service.image_asset, image_path)
            if asset.file_path is not None:
                return FileResponse(asset.file_path, media_type=asset.media_type)
            return Response(content=asset.content or b"", media_type=asset.media_type)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/images/{image_path:path}")
    async def delete_image(image_path: str):
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.delete_image, image_path)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/images/batch-delete")
    async def delete_images(payload: BatchDeletePayload):
        try:
            bind_service_loop()
            return await asyncio.to_thread(service.delete_images, payload.paths)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/events")
    async def events():
        bind_service_loop()
        return StreamingResponse(service.iter_events(), media_type="text/event-stream")

    return app
