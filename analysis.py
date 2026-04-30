"""Audio loading, windowing, and Distill-MOS scoring for stereo call recordings."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchaudio


TARGET_SR = 16000
WINDOW_SECONDS = 8.0
HOP_SECONDS = 2.0
SILENCE_RMS = 0.01
# A scoring window counts as voiced only if at least this fraction of its
# samples falls inside a Silero-VAD speech segment. Tuned so a window where
# the talker starts mid-window still scores, but pure pre-speech hiss does not.
MIN_SPEECH_OVERLAP_FRAC = 0.3
# Split the MOS forward pass into mini-batches so we can emit progress between them.
SCORE_MINIBATCH = 16

ProgressCb = Callable[[str, float], None]


class AudioValidationError(ValueError):
    """Raised when the uploaded audio fails validation (mono, corrupt, etc.)."""


@dataclass
class WindowResult:
    start_s: float
    rms: float
    mos: float | None  # None for silent windows
    silent: bool


@dataclass
class LegResult:
    label: str
    windows: list[WindowResult]
    mos_median: float | None
    mos_p10: float | None
    mos_p90: float | None
    pct_below_3: float | None
    worst_window_timestamp: float | None
    voiced_seconds: float

    def status(self) -> str:
        m = self.mos_median
        if m is None:
            return "No speech"
        if m >= 4.0:
            return "Excellent"
        if m >= 3.5:
            return "Good"
        if m >= 3.0:
            return "Fair"
        return "Poor"


@dataclass
class AnalysisResult:
    filename: str
    duration_seconds: float
    sample_rate_in: int
    leg_a: LegResult
    leg_b: LegResult | None  # None when the input is mono.


def load_wav(data: bytes) -> tuple[torch.Tensor, int]:
    """Load WAV bytes into a (channels, samples) float tensor and original sample rate.

    Accepts mono or stereo. Raises AudioValidationError on corrupt input.
    Channels beyond the second are dropped (FreeSWITCH stereo recordings are 2-channel).
    """
    try:
        waveform, sr = torchaudio.load(io.BytesIO(data))
    except Exception as e:  # torchaudio raises a variety of errors
        raise AudioValidationError(f"Could not decode audio file: {e}") from e

    if waveform.dim() != 2:
        raise AudioValidationError("Unexpected audio tensor shape.")
    if waveform.shape[0] == 0:
        raise AudioValidationError("Audio file contains no channels.")
    if waveform.shape[0] > 2:
        waveform = waveform[:2]

    return waveform.to(torch.float32), sr


def resample_if_needed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return waveform
    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
    return resampler(waveform)


def _window_indices(num_samples: int) -> list[int]:
    win = int(WINDOW_SECONDS * TARGET_SR)
    hop = int(HOP_SECONDS * TARGET_SR)
    if num_samples <= 0:
        return []
    if num_samples <= win:
        return [0]
    starts: list[int] = []
    s = 0
    while s + win <= num_samples:
        starts.append(s)
        s += hop
    # Pad-style last window so trailing audio is included.
    if starts[-1] + win < num_samples:
        starts.append(num_samples - win if num_samples >= win else 0)
    return starts


def _split_windows(leg: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    """Return (start_samples, padded_chunks_tensor[N, win_samples])."""
    win = int(WINDOW_SECONDS * TARGET_SR)
    n = leg.shape[-1]
    starts = _window_indices(n)
    chunks = []
    for s in starts:
        end = s + win
        if end <= n:
            chunk = leg[s:end]
        else:
            chunk = torch.zeros(win, dtype=leg.dtype)
            tail = leg[s:n]
            chunk[: tail.shape[0]] = tail
        chunks.append(chunk)
    if not chunks:
        return [], torch.zeros(0, win, dtype=leg.dtype)
    return starts, torch.stack(chunks, dim=0)


def _aggregate(windows: list[WindowResult]) -> dict:
    voiced = [w for w in windows if not w.silent and w.mos is not None]
    if not voiced:
        return dict(
            mos_median=None,
            mos_p10=None,
            mos_p90=None,
            pct_below_3=None,
            worst_window_timestamp=None,
            voiced_seconds=0.0,
        )
    mos = np.array([w.mos for w in voiced], dtype=np.float64)
    worst = min(voiced, key=lambda w: w.mos)
    return dict(
        mos_median=float(np.median(mos)),
        mos_p10=float(np.percentile(mos, 10)),
        mos_p90=float(np.percentile(mos, 90)),
        pct_below_3=float(np.mean(mos < 3.0)),
        worst_window_timestamp=float(worst.start_s),
        voiced_seconds=float(len(voiced) * HOP_SECONDS),
    )


def _vad_speech_mask(leg: torch.Tensor, starts: list[int], vad) -> np.ndarray:
    """Per-window boolean mask: True iff a scoring window overlaps speech.

    Runs Silero VAD once on the full leg to get speech sample-ranges, then
    computes per-window overlap fraction. Returns all-True if vad is None
    (graceful fallback so analysis still works without VAD loaded).
    """
    n_windows = len(starts)
    if n_windows == 0:
        return np.zeros(0, dtype=bool)
    if vad is None:
        return np.ones(n_windows, dtype=bool)

    from silero_vad import get_speech_timestamps

    win = int(WINDOW_SECONDS * TARGET_SR)
    speech = get_speech_timestamps(
        leg, vad, sampling_rate=TARGET_SR, return_seconds=False
    )
    if not speech:
        return np.zeros(n_windows, dtype=bool)

    mask = np.zeros(n_windows, dtype=bool)
    min_overlap = int(MIN_SPEECH_OVERLAP_FRAC * win)
    for i, s in enumerate(starts):
        end = s + win
        overlap = 0
        for seg in speech:
            seg_s, seg_e = seg["start"], seg["end"]
            if seg_e <= s:
                continue
            if seg_s >= end:
                break  # speech ranges are ordered; no later seg can overlap.
            overlap += min(end, seg_e) - max(s, seg_s)
            if overlap >= min_overlap:
                break
        mask[i] = overlap >= min_overlap
    return mask


def _score_leg(
    label: str,
    leg: torch.Tensor,
    model,
    vad,
    progress_cb: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.0, 1.0),
) -> LegResult:
    lo, hi = progress_range
    starts, chunks = _split_windows(leg)
    if chunks.numel() == 0:
        if progress_cb:
            progress_cb(f"No audio in {label}", hi)
        return LegResult(
            label=label,
            windows=[],
            mos_median=None,
            mos_p10=None,
            mos_p90=None,
            pct_below_3=None,
            worst_window_timestamp=None,
            voiced_seconds=0.0,
        )

    if progress_cb:
        progress_cb(f"Detecting speech ({label})…", lo)
    rms = torch.sqrt(torch.mean(chunks**2, dim=1)).cpu().numpy()
    speech_mask = _vad_speech_mask(leg, starts, vad)
    voiced_mask = speech_mask & (rms >= SILENCE_RMS)

    mos_per_window: dict[int, float] = {}
    if voiced_mask.any():
        voiced_idx = np.where(voiced_mask)[0]
        total = len(voiced_idx)
        for start in range(0, total, SCORE_MINIBATCH):
            sub = voiced_idx[start : start + SCORE_MINIBATCH]
            with torch.no_grad():
                preds = model(chunks[sub]).detach().cpu().numpy().reshape(-1)
            for i, score in zip(sub, preds):
                mos_per_window[int(i)] = float(score)
            if progress_cb:
                done = start + len(sub)
                frac = done / total
                progress_cb(
                    f"Scoring {label}… {done}/{total} windows",
                    lo + (hi - lo) * frac,
                )
    elif progress_cb:
        progress_cb(f"No voiced windows in {label}", hi)

    windows: list[WindowResult] = []
    for i, s in enumerate(starts):
        is_silent = not voiced_mask[i]
        windows.append(
            WindowResult(
                start_s=s / TARGET_SR,
                rms=float(rms[i]),
                mos=mos_per_window.get(i) if not is_silent else None,
                silent=bool(is_silent),
            )
        )

    agg = _aggregate(windows)
    return LegResult(label=label, windows=windows, **agg)


def analyze(
    data: bytes,
    filename: str,
    model,
    vad=None,
    progress_cb: ProgressCb | None = None,
) -> AnalysisResult:
    """End-to-end analysis of a mono or stereo WAV byte payload."""
    if progress_cb:
        progress_cb("Loading audio…", 0.0)
    waveform, sr = load_wav(data)
    duration = waveform.shape[-1] / sr
    waveform = resample_if_needed(waveform, sr)
    if progress_cb:
        progress_cb("Loaded audio", 0.03)

    is_stereo = waveform.shape[0] >= 2
    leg_a_t = waveform[0].contiguous()
    leg_a_label = "A-leg (caller)" if is_stereo else "Mono recording"
    a_range = (0.03, 0.50) if is_stereo else (0.03, 0.97)
    leg_a = _score_leg(leg_a_label, leg_a_t, model, vad, progress_cb, a_range)

    leg_b: LegResult | None = None
    if is_stereo:
        leg_b_t = waveform[1].contiguous()
        leg_b = _score_leg(
            "B-leg (callee)", leg_b_t, model, vad, progress_cb, (0.50, 0.97)
        )

    if progress_cb:
        progress_cb("Finalizing…", 0.98)

    return AnalysisResult(
        filename=filename,
        duration_seconds=duration,
        sample_rate_in=sr,
        leg_a=leg_a,
        leg_b=leg_b,
    )


def result_to_template_context(result: AnalysisResult) -> dict:
    """Shape an AnalysisResult for the Jinja results template + Chart.js JSON blob."""

    def leg_dict(leg: LegResult) -> dict:
        return {
            "label": leg.label,
            "mos_median": leg.mos_median,
            "mos_p10": leg.mos_p10,
            "mos_p90": leg.mos_p90,
            "pct_below_3": leg.pct_below_3,
            "worst_window_timestamp": leg.worst_window_timestamp,
            "voiced_seconds": leg.voiced_seconds,
            "status": leg.status(),
        }

    def chart_points(leg: LegResult) -> list[dict]:
        return [
            {
                "t": round(w.start_s, 3),
                "mos": (None if w.silent else round(w.mos, 3)),
                "silent": w.silent,
                "rms": round(w.rms, 5),
            }
            for w in leg.windows
        ]

    def worst_moments(leg: LegResult, n: int = 3) -> list[dict]:
        voiced = [w for w in leg.windows if not w.silent and w.mos is not None]
        voiced.sort(key=lambda w: w.mos)
        return [
            {"timestamp": round(w.start_s, 2), "mos": round(w.mos, 2)}
            for w in voiced[:n]
        ]

    chart_data = {"leg_a": chart_points(result.leg_a)}
    if result.leg_b is not None:
        chart_data["leg_b"] = chart_points(result.leg_b)

    return {
        "filename": result.filename,
        "duration_seconds": result.duration_seconds,
        "is_stereo": result.leg_b is not None,
        "leg_a": leg_dict(result.leg_a),
        "leg_b": leg_dict(result.leg_b) if result.leg_b is not None else None,
        "leg_a_worst": worst_moments(result.leg_a),
        "leg_b_worst": worst_moments(result.leg_b) if result.leg_b is not None else None,
        "chart_data": chart_data,
    }
