"""FastAPI server for stereo WAV call quality analysis using Distill-MOS."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import analysis

logger = logging.getLogger("call_quality")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
JOB_TTL_S = 600  # drop finished/abandoned jobs older than 10 min

templates = Jinja2Templates(directory="templates")


@dataclass
class JobState:
    job_id: str
    filename: str
    stage: str = "Queued…"
    percent: float = 0.0
    done: bool = False
    error: str | None = None
    error_detail: str | None = None
    result: analysis.AnalysisResult | None = None
    created_at: float = field(default_factory=monotonic)


JOBS: dict[str, JobState] = {}


def _cleanup_old_jobs() -> None:
    now = monotonic()
    stale = [jid for jid, j in JOBS.items() if now - j.created_at > JOB_TTL_S]
    for jid in stale:
        JOBS.pop(jid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import distillmos
    import torch
    from silero_vad import load_silero_vad

    logger.info("Loading Distill-MOS model...")
    model = distillmos.ConvTransformerSQAModel()
    model.eval()
    # Disable autograd globally for the loaded module's parameters.
    for p in model.parameters():
        p.requires_grad_(False)
    app.state.model = model
    app.state.torch = torch
    logger.info("Loading Silero VAD model...")
    app.state.vad = load_silero_vad()
    logger.info("Models loaded and ready.")
    yield
    app.state.model = None
    app.state.vad = None


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


def _render_results(request: Request, result: analysis.AnalysisResult) -> HTMLResponse:
    ctx = analysis.result_to_template_context(result)
    ctx["request"] = request
    ctx["analyzed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ctx["chart_data_json"] = json.dumps(ctx["chart_data"])
    return templates.TemplateResponse("results.html", ctx)


def _render_progress(request: Request, job: JobState) -> HTMLResponse:
    return templates.TemplateResponse(
        "_progress.html", {"request": request, "job": job}
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_endpoint(request: Request, file: UploadFile = File(...)):
    _cleanup_old_jobs()
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

    job = JobState(job_id=uuid.uuid4().hex, filename=filename)
    JOBS[job.job_id] = job
    vad = request.app.state.vad

    def _cb(stage: str, frac: float) -> None:
        # Called from the worker thread; simple field writes are safe enough
        # for the polling reader that just reads them.
        job.stage = stage
        pct = max(0.0, min(100.0, frac * 100.0))
        if pct > job.percent:
            job.percent = pct

    async def _run() -> None:
        try:
            result = await asyncio.to_thread(
                analysis.analyze, data, filename, model, vad, _cb
            )
            job.result = result
            job.stage = "Done"
        except analysis.AudioValidationError as e:
            job.error = "Could not analyze recording"
            job.error_detail = str(e)
            job.stage = "Failed"
        except Exception as e:  # last-resort safeguard
            logger.exception("Analysis failed")
            job.error = "Analysis failed"
            job.error_detail = f"An unexpected error occurred: {e}"
            job.stage = "Failed"
        finally:
            job.percent = 100.0
            job.done = True

    asyncio.create_task(_run())
    return _render_progress(request, job)


@app.get("/progress/{job_id}", response_class=HTMLResponse)
async def progress_endpoint(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return _render_error(
            request,
            "Job not found",
            "The analysis job has expired or was never started. Please re-upload.",
        )
    if job.done:
        if job.error:
            return _render_error(request, job.error, job.error_detail)
        if job.result is not None:
            return _render_results(request, job.result)
    return _render_progress(request, job)
