"""Local upload API for the frozen CF detector and provenance evidence."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.detector import CFDetector
from backend.pipeline import AnalysisPipeline, InvalidImage, PipelineBusy, PipelineLimits, UploadTooLarge


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class UploadBodyLimit:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/analyze":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({"detail": "Upload exceeds the request limit."}, status_code=413)(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    raise HTTPException(413, "Upload exceeds the request limit.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(pipeline_factory=None, *, limits: PipelineLimits | None = None, cors_origins=None) -> FastAPI:
    limits = limits or PipelineLimits()

    @asynccontextmanager
    async def lifespan(app):
        if pipeline_factory:
            app.state.pipeline = pipeline_factory()
        else:
            artifacts = Path(os.environ.get("AI_SHIELD_ARTIFACT_DIR", REPO_ROOT / ".local/backend/artifacts"))
            anchors_path = os.environ.get("AI_SHIELD_TRUST_ANCHORS")
            detector = CFDetector(
                artifacts, device=os.environ.get("AI_SHIELD_DEVICE", "cpu"),
                threads=int(os.environ.get("AI_SHIELD_TORCH_THREADS", "2")),
            )
            app.state.pipeline = AnalysisPipeline(
                detector, limits=limits, exiftool=os.environ.get("EXIFTOOL_PATH"),
                trust_anchors_pem=Path(anchors_path).read_text(encoding="utf-8") if anchors_path else None,
            )
        try:
            yield
        finally:
            app.state.pipeline = None

    app = FastAPI(title="AI Shield API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(UploadBodyLimit, max_bytes=limits.max_bytes + 64 * 1024)
    origins = cors_origins if cors_origins is not None else [
        value.strip() for value in os.environ.get("AI_SHIELD_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if value.strip()
    ]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.get("/health")
    def health():
        return {"status": "ready", "detector": app.state.pipeline.detector.identity}

    @app.post("/analyze")
    def analyze(file: UploadFile, include_gps: bool = False):
        try:
            return app.state.pipeline.analyze(file.file, file.filename, include_gps=include_gps)
        except UploadTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except InvalidImage as exc:
            raise HTTPException(422, str(exc)) from exc
        except PipelineBusy as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "2"}) from exc
        except Exception as exc:
            LOGGER.exception("Image analysis failed")
            raise HTTPException(500, "Image analysis could not complete.") from exc
        finally:
            file.file.close()

    return app


app = create_app()
