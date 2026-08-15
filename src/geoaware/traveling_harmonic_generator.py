"""Independent analytic generator for the phase-wave R2 decision experiment.

The learner is deliberately not imported here.  In particular, the irrational
generator frequencies below are never used to initialize the learner's broad,
trainable frequency dictionary.
"""

from __future__ import annotations

import math

import numpy as np


GENERATOR_FREQUENCIES_HZ = (math.sqrt(13.0), math.sqrt(41.0))


def traveling_harmonic_phase(
    travel_time: np.ndarray,
    times: np.ndarray,
    source_xy: np.ndarray,
) -> np.ndarray:
    """Return a two-band field that depends on space/time only through τ-t."""
    tau_minus_t = (
        np.asarray(travel_time, dtype=np.float64)[None, :]
        - np.asarray(times, dtype=np.float64)[:, None]
    )
    source = np.asarray(source_xy, dtype=np.float64)
    source_phase = 0.31 * source[0] - 0.23 * source[1]
    f1, f2 = GENERATOR_FREQUENCIES_HZ
    return (
        np.cos(2 * math.pi * f1 * tau_minus_t + source_phase)
        + 0.42 * np.sin(2 * math.pi * f2 * tau_minus_t - 0.37 + source_phase)
    ).astype(np.float32)


def generate_traveling_harmonic(
    travel_time: np.ndarray,
    times: np.ndarray,
    coordinates: np.ndarray,
    source_xy: np.ndarray,
) -> np.ndarray:
    """Generate an attenuated source-conditioned traveling harmonic mixture.

    The envelope is spatial-only.  Therefore every carrier remains a genuine
    traveling harmonic rather than the standing-wave or moving-envelope cases
    that confounded earlier experiments.
    """
    tau = np.asarray(travel_time, dtype=np.float64)
    xy = np.asarray(coordinates, dtype=np.float64)
    source = np.asarray(source_xy, dtype=np.float64)
    envelope = np.exp(-0.08 * tau) / (1.0 + 0.12 * tau)
    envelope *= (
        1.0
        + 0.10 * np.sin(1.7 * xy[:, 0] + 0.4 * source[0])
        * np.cos(1.3 * xy[:, 1] - 0.2 * source[1])
    )
    return (traveling_harmonic_phase(tau, times, source) * envelope[None, :]).astype(
        np.float32
    )
