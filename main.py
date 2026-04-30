"""FastAPI server for stereo WAV call quality analysis using Distill-MOS."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import analysis

logger = logging.getLogger("call_quality")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import distillmos
    import torch

    logger.info("Loading Distill-MOS model...")
    model = distillmos.ConvTransformerSQAModel()
    model.eval()
    # Disable autograd globally for the loaded module's parameters.
    for p in model.parameters():
        p.requires_grad_(False)
    app.state.model = model
    app.state.torch = torch
    logger.info("Model loaded and ready.")
    yield
    app.state.model = None


app = FastAPI(title="Call Quality Analyzer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health(request: Request):
    ready = getattr(request.app.state, "model", None) is not None
    return {"status": "ok"} if ready else {"status": "loading"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


def _render_error(request: Request, message: str, detail: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "message": message, "detail": detail},
        status_code=200,  # HTMX swap target — keep 200 so swap fires.
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_endpoint(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "upload.wav"

    # Quick MIME / extension sanity check.
    lower = filename.lower()
    if not (lower.endswith(".wav") or (file.content_type or "").startswith("audio/")):
        return _render_error(
            request,
            "Unsupported file type",
            "Please upload a stereo WAV file recorded by FreeSWITCH.",
        )

    data = await file.read()
    if len(data) == 0:
        return _render_error(request, "Empty file", "The uploaded file contains no data.")
    if len(data) > MAX_UPLOAD_BYTES:
        return _render_error(
            request,
            "File too large",
            f"Uploads must be 100 MB or smaller. Yours was {len(data) / (1024 * 1024):.1f} MB.",
        )

    model = request.app.state.model
    if model is None:
        return _render_error(
            request,
            "Model not ready",
            "The MOS model is still loading. Please retry in a moment.",
        )

    try:
        result = analysis.analyze(data, filename=filename, model=model)
    except analysis.AudioValidationError as e:
        return _render_error(request, "Could not analyze recording", str(e))
    except Exception as e:  # last-resort safeguard
        logger.exception("Analysis failed")
        return _render_error(
            request,
            "Analysis failed",
            f"An unexpected error occurred: {e}",
        )

    ctx = analysis.result_to_template_context(result)
    ctx["request"] = request
    ctx["analyzed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ctx["chart_data_json"] = json.dumps(ctx["chart_data"])
    return templates.TemplateResponse("results.html", ctx)
