"""Frame-aligned prosody supervision for the 16 kHz WavLM tokenizer."""

import numpy as np


SAMPLE_RATE = 16000
TOKEN_HOP = 320
FEATURE_WINDOW = 400
TARGET_COLUMNS = 6


def extract_prosody(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return [F0 st, energy dB, voiced, pitch mask, voicing mask, frame mask]."""
    import pyworld

    signal = np.ascontiguousarray(waveform, dtype=np.float64)
    if (sample_rate != SAMPLE_RATE or signal.ndim != 1
            or signal.size < FEATURE_WINDOW or not np.isfinite(signal).all()):
        raise ValueError('Prosody extraction requires a finite mono 16 kHz waveform')

    raw, times = pyworld.dio(signal, SAMPLE_RATE, frame_period=10.0,
                             f0_floor=65.0, f0_ceil=600.0)
    f0 = pyworld.stonemask(signal, raw, times, SAMPLE_RATE)
    f0_voiced = np.isfinite(f0) & (f0 >= 65.0) & (f0 <= 600.0)
    reliable = f0_voiced & np.isfinite(raw) & (raw > 0)
    reliable &= np.abs(np.log2(np.maximum(f0, 1.) / np.maximum(raw, 1.))) <= 0.5
    unvoiced = (raw == 0) & (f0 == 0)

    starts = np.arange(0, signal.size, TOKEN_HOP)
    ends = np.minimum(starts + FEATURE_WINDOW, signal.size)
    centers = (starts + (FEATURE_WINDOW - 1) / 2) / SAMPLE_RATE
    nearest = np.clip(np.floor(centers / 0.010 + 0.5).astype(int), 0, len(times) - 1)
    # S3PRL can extend the final frame to ceil(N / 320); mask positions without
    # a complete convolution window along with any dataloader padding.
    frame_mask = starts + FEATURE_WINDOW <= signal.size
    pitch_mask = frame_mask & reliable[nearest]
    voicing_mask = frame_mask & (reliable | unvoiced)[nearest]

    f0_target = np.zeros(starts.size)
    if reliable.any():
        f0_target[pitch_mask] = 12 * np.log2(f0[nearest][pitch_mask] / np.median(f0[reliable]))
    power_sum = np.concatenate(([0.0], np.cumsum(np.square(signal))))
    energy = 10 * np.log10(np.maximum((power_sum[ends] - power_sum[starts]) / (ends - starts), 1e-10))
    active = frame_mask & (energy >= energy[frame_mask].max() - 40.0)
    energy -= np.median(energy[active])
    return np.column_stack((f0_target, energy, reliable[nearest], pitch_mask,
                            voicing_mask, frame_mask)).astype(np.float32)
