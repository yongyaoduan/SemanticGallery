from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from desktop_runtime.api import build_app
from desktop_runtime.index_store import IndexStore
from desktop_runtime.mlx_encoder import MLXEmbeddingEncoder
from desktop_runtime.service import DesktopService
from desktop_runtime.stage2_jobs import ScriptStage2Runner, Stage2Job


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the SemanticGallery desktop sidecar.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=36168)
    parser.add_argument("--workspace-root", default=Path(__file__).resolve().parents[1].as_posix())
    parser.add_argument("--index-db", default=None)
    return parser.parse_args()


def default_index_db_path(workspace_root: Path) -> Path:
    runtime_root = workspace_root.expanduser().resolve()
    support_root = runtime_root.parent if runtime_root.name == "runtime" else runtime_root
    return support_root / "index.sqlite3"


def build_service(workspace_root: Path, index_db_path: Path | None = None) -> DesktopService:
    runtime_root = workspace_root.expanduser().resolve()
    db_path = index_db_path.expanduser().resolve() if index_db_path else default_index_db_path(runtime_root)
    store = IndexStore.connect(db_path)
    store.migrate()

    def encoder_loader(folder_path: Path, encoder_signature: str):
        return MLXEmbeddingEncoder(runtime_root, folder_path, encoder_signature)

    service = DesktopService(
        store=store,
        encoder_loader=encoder_loader,
        thumbnails_dir=db_path.parent / "thumbs",
    )
    service.stage2_job = Stage2Job(
        ScriptStage2Runner(
            runtime_root,
            emit=lambda line: service.publish_event("stage2-log", {"line": line}),
        )
    )
    service.setup_status = "ready"
    service.last_task_message = "Choose a folder to build the local index."
    return service


def main():
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    index_db_path = Path(args.index_db).expanduser().resolve() if args.index_db else None

    service = build_service(workspace_root, index_db_path=index_db_path)
    async def on_startup():
        service.start_watch_loop()
        print(f"READY http://{args.host}:{args.port}", flush=True)

    app = build_app(service, on_startup=on_startup)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
