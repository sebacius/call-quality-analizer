# Call Quality Analyzer

A small FastAPI + HTMX web app that scores **FreeSWITCH stereo WAV recordings**
with Microsoft's [Distill-MOS](https://github.com/microsoft/Distill-MOS) speech
quality model. Channel 0 is treated as the **A-leg** (caller), channel 1 as the
**B-leg** (callee). Each leg is windowed (8 s window / 2 s hop), silence-gated
(RMS &lt; 0.01), and scored independently.

The result page shows median / P10 / P90 MOS per leg, the fraction of windows
below a 3.0 MOS, voiced duration, a quality-over-time line chart, and the worst
moments per leg with timestamps.

## Install

Managed with [`uv`](https://docs.astral.sh/uv/) — installs Python, creates the
project venv, and resolves the lockfile in one step.

```bash
# One-time, if you don't have uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Inside this repo:
uv sync
```

> The first model invocation downloads the Distill-MOS weights automatically
> (a few hundred MB). Subsequent boots are fast.

## Run

```bash
uv run uvicorn main:app --reload
```

Open http://127.0.0.1:8000 in your browser. No need to activate a venv — `uv
run` resolves and uses the project environment automatically.

- `GET /` &mdash; upload UI
- `GET /health` &mdash; returns `{"status":"ok"}` once the model is loaded
- `POST /analyze` &mdash; multipart upload of a stereo WAV; returns an HTMX
  partial that swaps into `#results`

## Notes

- Uploads are processed entirely in memory and discarded; nothing is written to
  disk.
- Maximum upload size is 100 MB. Anything larger renders a friendly error card.
- The Distill-MOS model is loaded **once** at server startup via FastAPI's
  `lifespan` and reused for every request. Inference runs under
  `torch.no_grad()` with the model in `eval()` mode.
