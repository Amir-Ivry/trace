"""CPU-only audio functions; no inference, scheduling, or implicit file writes.

Extracted/re-expressed from the validated v3 prepare.py additive operation.
The full machinery event can overlap neighboring words: it is not word-gated.
All coordinates accepted by these functions refer to the supplied speech array.
"""
from pathlib import Path
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_mono(path):
    """Decode PCM WAV; integer values follow SciPy's left-aligned representation."""
    rate, raw = wavfile.read(Path(path))
    if raw.dtype.kind == "i":
        samples = raw.astype(np.float64) / (2 ** (raw.dtype.itemsize * 8 - 1))
    elif raw.dtype.kind == "u" and raw.dtype.itemsize == 1:
        samples = (raw.astype(np.float64) - 128.0) / 128.0
    elif raw.dtype.kind == "f":
        samples = raw.astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV dtype {raw.dtype}")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono/stereo WAV")
    return samples, int(rate)


def machinery_at_rate(path, sample_rate=16000):
    """Whole released event, arithmetic stereo mean, fixed polyphase resampling.

    No cropping, looping, normalization, denoising, fading, or time stretching.
    The released source itself had creator-reported mastering/beat cropping.
    """
    samples, source_rate = read_mono(path)
    divisor = math.gcd(source_rate, sample_rate)
    if source_rate != sample_rate:
        samples = resample_poly(
            samples, sample_rate // divisor, source_rate // divisor,
            window=("kaiser", 5.0), padtype="constant"
        )
    return np.asarray(samples, dtype=np.float64)


def additive_over_word(speech, noise, word_start, word_end, snr_db=-10.0):
    """Add full noise event, centering cumulative-energy median on word midpoint.

    SNR = 10 log10(mean(speech[word]^2) / mean(scaled_noise[word]^2)).
    Noise is scaled using only the word interval, not whole-utterance energy.
    Returns an UNCLIPPED float64 mixture and audit values. Apply one common
    anti-clipping gain to all comparison conditions with common_anticlip_gain.
    No padding is added implicitly; an event extending outside speech is rejected.
    """
    x, n = np.asarray(speech, np.float64), np.asarray(noise, np.float64)
    a, b = int(word_start), int(word_end)
    if x.ndim != 1 or n.ndim != 1 or not (0 <= a < b <= len(x)):
        raise ValueError("Invalid mono input or word interval")
    if not np.isfinite(x).all() or not np.isfinite(n).all():
        raise ValueError("Nonfinite input")
    energy = n ** 2
    total = float(energy.sum())
    if total <= 0:
        raise ValueError("Noise has zero energy")
    median = int(np.searchsorted(np.cumsum(energy), .5 * total))
    start = round((a + b) / 2 - median)
    if start < 0 or start + len(n) > len(x):
        raise ValueError("Full event exceeds speech extent; explicit shared padding required")
    placed = np.zeros_like(x)
    placed[start:start + len(n)] = n
    speech_power = float(np.mean(x[a:b] ** 2))
    noise_power = float(np.mean(placed[a:b] ** 2))
    if speech_power <= 0 or noise_power <= 0:
        raise ValueError("Calibration interval must have speech and noise energy")
    scale = math.sqrt(speech_power / noise_power) * 10 ** (-float(snr_db) / 20)
    scaled = placed * scale
    result = x + scaled
    measured = 10 * math.log10(speech_power / float(np.mean(scaled[a:b] ** 2)))
    return result, {
        "word_samples_half_open": [a, b],
        "event_samples_half_open": [start, start + len(n)],
        "energy_median_sample": median,
        "noise_gain_before_common_attenuation": scale,
        "requested_local_snr_db": float(snr_db),
        "measured_local_snr_db": measured,
        "numeral_fraction_of_event_energy": float((placed[a:b] ** 2).sum() / total),
        "unscaled_mixture_peak": float(np.abs(result).max()),
        "boundary_convention": "zero-based, start inclusive, end exclusive",
    }


def silence_region(speech, word_start, word_end):
    """Set exactly the supplied interval to zero without changing duration.

    No fades or extension into adjacent samples. Automated word boundaries do
    not prove that this removes every phonetic cue; listening remains necessary.
    """
    result = np.asarray(speech, dtype=np.float64).copy()
    if not (0 <= word_start < word_end <= len(result)):
        raise ValueError("Invalid silence interval")
    result[word_start:word_end] = 0.0
    return result


def common_anticlip_gain(conditions, peak_limit=.989):
    """Apply ONE attenuation to every condition; no clipping/normalization per file.

    The untouched source asset stays unchanged. Returned baseline audio may be
    gain-scaled and must be labeled accordingly. SNR is unchanged by common gain.
    """
    if not conditions or not (0 < peak_limit < 1):
        raise ValueError("Conditions and peak limit required")
    values = {k: np.asarray(v, dtype=np.float64) for k, v in conditions.items()}
    if any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Invalid conditions")
    peak = max(float(np.abs(v).max()) for v in values.values())
    gain = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    return {k: (v * gain).astype(np.float32) for k, v in values.items()}, gain
