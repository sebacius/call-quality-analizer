"""Unit tests for `compute_turn_metrics` in analysis.py.

Targeted at the pure interval math; does not load audio, models, or VAD.
"""

from __future__ import annotations

import math

import pytest

from analysis import compute_turn_metrics


def approx(x: float | None, expected: float, tol: float = 1e-6) -> bool:
    return x is not None and math.isclose(x, expected, abs_tol=tol)


def test_pure_alternation_no_overlap():
    # A speaks 0-4, gap, B replies 5-8, gap, A replies 10-12. Duration 15s.
    a = [(0.0, 4.0), (10.0, 12.0)]
    b = [(5.0, 8.0)]
    cm = compute_turn_metrics(a, b, duration_s=15.0)

    assert approx(cm.a_talk_s, 6.0)
    assert approx(cm.b_talk_s, 3.0)
    assert approx(cm.overlap_s, 0.0)
    # Silence = 15 - (4 + 3 + 2) = 6
    assert approx(cm.mutual_silence_s, 6.0)
    assert cm.turn_count == 2
    # A→B gap: 5 - 4 = 1.0
    assert approx(cm.median_response_latency_a_to_b_s, 1.0)
    # B→A gap: 10 - 8 = 2.0
    assert approx(cm.median_response_latency_b_to_a_s, 2.0)
    assert approx(cm.max_response_latency_s, 2.0)


def test_overlap_and_bargein():
    # A speaks 0-5, B barges in at 3 and runs to 8. Duration 10s.
    a = [(0.0, 5.0)]
    b = [(3.0, 8.0)]
    cm = compute_turn_metrics(a, b, duration_s=10.0)

    assert approx(cm.a_talk_s, 5.0)
    assert approx(cm.b_talk_s, 5.0)
    # Union = (0,8) length 8 → overlap = 5+5-8 = 2
    assert approx(cm.overlap_s, 2.0)
    assert approx(cm.mutual_silence_s, 2.0)  # 10 - 8
    # One handoff (A→B), latency clamped to 0 because of overlap.
    assert cm.turn_count == 1
    assert approx(cm.median_response_latency_a_to_b_s, 0.0)
    assert cm.median_response_latency_b_to_a_s is None


def test_long_silence():
    # Single A turn, then 30s silence, then a B turn. Duration 40s.
    a = [(0.0, 2.0)]
    b = [(35.0, 38.0)]
    cm = compute_turn_metrics(a, b, duration_s=40.0)

    assert approx(cm.mutual_silence_s, 35.0)
    assert cm.turn_count == 1
    assert approx(cm.median_response_latency_a_to_b_s, 33.0)
    assert approx(cm.max_response_latency_s, 33.0)


def test_intra_turn_pauses_dont_count_as_turns():
    # A has two segments separated by a 1s pause; B replies once. One turn only.
    a = [(0.0, 3.0), (4.0, 6.0)]
    b = [(7.0, 9.0)]
    cm = compute_turn_metrics(a, b, duration_s=10.0)

    assert cm.turn_count == 1
    # A→B gap: 7 - 6 = 1.0 (uses the *last* end of A's run, not the first segment)
    assert approx(cm.median_response_latency_a_to_b_s, 1.0)


def test_mono_only_silence_and_talk():
    a = [(0.0, 2.0), (5.0, 8.0)]
    cm = compute_turn_metrics(a, None, duration_s=10.0)

    assert approx(cm.a_talk_s, 5.0)
    assert approx(cm.mutual_silence_s, 5.0)
    assert cm.b_talk_s is None
    assert cm.overlap_s is None
    assert cm.turn_count is None
    assert cm.median_response_latency_a_to_b_s is None
    assert cm.median_response_latency_b_to_a_s is None
    assert cm.max_response_latency_s is None


def test_empty_segments_all_silence():
    cm = compute_turn_metrics([], [], duration_s=12.0)

    assert approx(cm.a_talk_s, 0.0)
    assert approx(cm.b_talk_s, 0.0)
    assert approx(cm.overlap_s, 0.0)
    assert approx(cm.mutual_silence_s, 12.0)
    assert cm.turn_count == 0
    assert cm.median_response_latency_a_to_b_s is None
    assert cm.median_response_latency_b_to_a_s is None
    assert cm.max_response_latency_s is None


def test_clipping_to_duration():
    # Segments running past duration should be clipped, not counted past the end.
    a = [(0.0, 100.0)]  # claims to run past duration
    b = [(2.0, 4.0)]
    cm = compute_turn_metrics(a, b, duration_s=5.0)

    assert approx(cm.a_talk_s, 5.0)
    assert approx(cm.b_talk_s, 2.0)
    assert approx(cm.mutual_silence_s, 0.0)


def test_within_leg_overlap_merged():
    # If a leg accidentally has overlapping segments they merge (no double-count).
    a = [(0.0, 3.0), (2.0, 5.0)]
    cm = compute_turn_metrics(a, [], duration_s=10.0)
    assert approx(cm.a_talk_s, 5.0)


def test_zero_duration_safe():
    cm = compute_turn_metrics([], [], duration_s=0.0)
    assert approx(cm.mutual_silence_s, 0.0)
    assert approx(cm.a_talk_s, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
