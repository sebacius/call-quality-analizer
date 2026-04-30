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
    speech_segments: list[tuple[float, float]]  # (start_s, end_s) from Silero VAD

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
class ConversationMetrics:
    """Stereo-derived flow metrics: silence, talk, overlap, turn-taking latency.

    For mono recordings only `mutual_silence_s` and `a_talk_s` are populated;
    inter-leg fields are None since there is no second speaker channel.
    """
    mutual_silence_s: float
    a_talk_s: float
    b_talk_s: float | None
    overlap_s: float | None
    turn_count: int | None
    median_response_latency_a_to_b_s: float | None
    median_response_latency_b_to_a_s: float | None
    max_response_latency_s: float | None


@dataclass
class AnalysisResult:
    filename: str
    duration_seconds: float
    sample_rate_in: int
    leg_a: LegResult
    leg_b: LegResult | None  # None when the input is mono.
    conversation: ConversationMetrics


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


def _vad_speech_mask(
    leg: torch.Tensor, starts: list[int], vad
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Per-window boolean mask plus the underlying VAD speech segments in seconds.

    Runs Silero VAD once on the full leg to get speech sample-ranges, then
    computes per-window overlap fraction. Returns all-True (and an empty
    segment list) if vad is None — graceful fallback so analysis still works
    without VAD loaded.
    """
    n_windows = len(starts)
    if vad is None:
        return np.ones(n_windows, dtype=bool), []

    from silero_vad import get_speech_timestamps

    win = int(WINDOW_SECONDS * TARGET_SR)
    speech = get_speech_timestamps(
        leg, vad, sampling_rate=TARGET_SR, return_seconds=False
    )
    segments_s: list[tuple[float, float]] = [
        (seg["start"] / TARGET_SR, seg["end"] / TARGET_SR) for seg in speech
    ]

    if n_windows == 0 or not speech:
        return np.zeros(n_windows, dtype=bool), segments_s

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
    return mask, segments_s


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
            speech_segments=[],
        )

    if progress_cb:
        progress_cb(f"Detecting speech ({label})…", lo)
    rms = torch.sqrt(torch.mean(chunks**2, dim=1)).cpu().numpy()
    speech_mask, speech_segments = _vad_speech_mask(leg, starts, vad)
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
    return LegResult(
        label=label, windows=windows, speech_segments=speech_segments, **agg
    )


def _merge_intervals(
    segs: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Clip to [0, duration_s], drop empties, merge overlapping/adjacent intervals."""
    cleaned: list[tuple[float, float]] = []
    for s, e in segs:
        s = max(0.0, min(float(s), duration_s))
        e = max(0.0, min(float(e), duration_s))
        if e > s:
            cleaned.append((s, e))
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[tuple[float, float]] = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _interval_union_length(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total length of the union of two sets of intervals (each pre-merged)."""
    events: list[tuple[float, float]] = sorted(a + b)
    if not events:
        return 0.0
    total = 0.0
    cur_s, cur_e = events[0]
    for s, e in events[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def compute_turn_metrics(
    a_segments: list[tuple[float, float]],
    b_segments: list[tuple[float, float]] | None,
    duration_s: float,
) -> ConversationMetrics:
    """Compute silence/talk/overlap/turn metrics from per-leg VAD segments.

    For mono input pass `b_segments=None`; only `mutual_silence_s` and
    `a_talk_s` will be populated.

    Turn-taking heuristic: walk all segments sorted by start time, treating
    consecutive same-leg segments as one turn. A handoff is when the new
    segment's leg differs from the previous one; the latency is
    `max(0, this.start - prev.end)`. Negative gaps (barge-in / overlap) clamp
    to 0 here and surface separately as `overlap_s`.
    """
    a = _merge_intervals(a_segments, duration_s)
    a_talk = sum(e - s for s, e in a)

    if b_segments is None:
        union = sum(e - s for s, e in a)
        return ConversationMetrics(
            mutual_silence_s=max(0.0, duration_s - union),
            a_talk_s=a_talk,
            b_talk_s=None,
            overlap_s=None,
            turn_count=None,
            median_response_latency_a_to_b_s=None,
            median_response_latency_b_to_a_s=None,
            max_response_latency_s=None,
        )

    b = _merge_intervals(b_segments, duration_s)
    b_talk = sum(e - s for s, e in b)
    union = _interval_union_length(a, b)
    overlap = max(0.0, a_talk + b_talk - union)
    mutual_silence = max(0.0, duration_s - union)

    events: list[tuple[float, float, str]] = (
        [(s, e, "a") for s, e in a] + [(s, e, "b") for s, e in b]
    )
    events.sort()

    a_to_b: list[float] = []
    b_to_a: list[float] = []
    turn_count = 0
    prev_leg: str | None = None
    prev_end: float = 0.0
    for s, e, leg in events:
        if prev_leg is None:
            prev_leg, prev_end = leg, e
            continue
        if leg != prev_leg:
            turn_count += 1
            gap = max(0.0, s - prev_end)
            (a_to_b if prev_leg == "a" else b_to_a).append(gap)
            prev_leg = leg
            prev_end = e
        else:
            prev_end = max(prev_end, e)

    def _median(xs: list[float]) -> float | None:
        return float(np.median(xs)) if xs else None

    all_latencies = a_to_b + b_to_a
    return ConversationMetrics(
        mutual_silence_s=mutual_silence,
        a_talk_s=a_talk,
        b_talk_s=b_talk,
        overlap_s=overlap,
        turn_count=turn_count,
        median_response_latency_a_to_b_s=_median(a_to_b),
        median_response_latency_b_to_a_s=_median(b_to_a),
        max_response_latency_s=float(max(all_latencies)) if all_latencies else None,
    )


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

    conversation = compute_turn_metrics(
        leg_a.speech_segments,
        leg_b.speech_segments if leg_b is not None else None,
        duration,
    )

    return AnalysisResult(
        filename=filename,
        duration_seconds=duration,
        sample_rate_in=sr,
        leg_a=leg_a,
        leg_b=leg_b,
        conversation=conversation,
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

    # Plot/report each window at its center so the dot sits at the midpoint
    # of the 8s of audio it scored — symmetric on both sides of the playhead.
    score_offset = WINDOW_SECONDS / 2

    def chart_points(leg: LegResult) -> list[dict]:
        return [
            {
                "t": round(w.start_s + score_offset, 3),
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
            {"timestamp": round(w.start_s + score_offset, 2), "mos": round(w.mos, 2)}
            for w in voiced[:n]
        ]

    chart_data = {"leg_a": chart_points(result.leg_a)}
    if result.leg_b is not None:
        chart_data["leg_b"] = chart_points(result.leg_b)

    cm = result.conversation
    dur = result.duration_seconds or 0.0

    def _frac(x: float | None) -> float | None:
        if x is None or dur <= 0:
            return None
        return x / dur

    conversation = {
        "mutual_silence_s": cm.mutual_silence_s,
        "mutual_silence_frac": _frac(cm.mutual_silence_s),
        "a_talk_s": cm.a_talk_s,
        "a_talk_frac": _frac(cm.a_talk_s),
        "b_talk_s": cm.b_talk_s,
        "b_talk_frac": _frac(cm.b_talk_s),
        "overlap_s": cm.overlap_s,
        "overlap_frac": _frac(cm.overlap_s),
        "turn_count": cm.turn_count,
        "median_response_latency_a_to_b_s": cm.median_response_latency_a_to_b_s,
        "median_response_latency_b_to_a_s": cm.median_response_latency_b_to_a_s,
        "max_response_latency_s": cm.max_response_latency_s,
    }

    return {
        "filename": result.filename,
        "duration_seconds": result.duration_seconds,
        "is_stereo": result.leg_b is not None,
        "leg_a": leg_dict(result.leg_a),
        "leg_b": leg_dict(result.leg_b) if result.leg_b is not None else None,
        "leg_a_worst": worst_moments(result.leg_a),
        "leg_b_worst": worst_moments(result.leg_b) if result.leg_b is not None else None,
        "chart_data": chart_data,
        "conversation": conversation,
    }
