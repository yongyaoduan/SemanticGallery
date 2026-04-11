from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from desktop_runtime.service import DesktopService, DesktopServiceError
from desktop_runtime.stage2_jobs import Stage2Error


class FolderSelectionPayload(BaseModel):
    folderPath: str


def build_app(service: DesktopService | object | None = None) -> FastAPI:
    service = service or DesktopService()
    app = FastAPI(title="SemanticGallery Desktop Sidecar", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/runtime/status")
    async def runtime_status():
        return service.runtime_status()

    @app.post("/api/folders/select")
    async def select_folder(payload: FolderSelectionPayload):
        try:
            return service.set_active_folder(payload.folderPath)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/folders/refresh")
    async def refresh_folder():
        try:
            return service.refresh_active_folder(lightweight=False)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage2/run")
    async def run_stage2():
        try:
            return service.run_stage2_for_active_folder()
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/search")
    async def search(query: str = Query("", alias="q"), limit: int = Query(25, ge=1, le=100)):
        try:
            return service.search_text(query, limit)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/metadata/{image_path:path}")
    async def metadata(image_path: str):
        try:
            return service.metadata(image_path)
        except (DesktopServiceError, Stage2Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/events")
    async def events():
        return StreamingResponse(service.iter_events(), media_type="text/event-stream")

    return app
