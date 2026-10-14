"""Prepare the three companion WAVs on CPU. No model or scheduler imports."""
import argparse
import io
import hashlib
import json
from pathlib import Path
import platform
import sys
import wave

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "source" / "TRACE_COMPANION_TRANSFER"
sys.path.insert(0, str(TRANSFER / "code"))
from audio_recipe import read_mono, machinery_at_rate, additive_over_word, silence_region, common_anticlip_gain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    output = Path(output).resolve()
    if output == TRANSFER.resolve() or TRANSFER.resolve() in output.parents:
        raise ValueError("The immutable transfer cannot be an output directory")
    speech_meta = json.loads((TRANSFER / "metadata/speech.json").read_text())
    noise_meta = json.loads((TRANSFER / "metadata/noise.json").read_text())
    speech_path = TRANSFER / speech_meta["asset_path"]
    noise_path = TRANSFER / "assets/noise/95_holman-nc4_3-start1_1beat.wav"
    if sha(speech_path) != speech_meta["sha256"] or sha(noise_path) != noise_meta["sha256"]:
        raise ValueError("Untouched source asset checksum mismatch")
    speech, rate = read_mono(speech_path)
    if rate != 16000 or len(speech) != 28989:
        raise ValueError("Unexpected original speech format")
    a = speech_meta["nine_boundary"]["start_sample_in_original"]
    b = speech_meta["nine_boundary"]["end_sample_in_original"]
    noise = machinery_at_rate(noise_path, rate)
    mixture, audit = additive_over_word(speech, noise, a, b, -10.0)
    inputs = {"reference": speech, "machinery": mixture,
              "segment_silenced": silence_region(speech, a, b)}
    floats, gain = common_anticlip_gain(inputs)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    encoded = {}
    files = {}
    for name, samples in floats.items():
        pcm = np.rint(samples.astype(np.float64) * 32768).astype("<i2")
        if np.any(np.abs(samples) >= 1):
            raise ValueError("Samples must be attenuated before PCM16 conversion")
        path = audio_dir / (name + ".wav")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(pcm.tobytes())
        data = buffer.getvalue()
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite differing audio evidence: {path}")
        if not path.exists():
            path.write_bytes(data)
        encoded[name] = pcm
        files[name] = {"path": "audio/" + path.name, "sha256": sha(path),
                       "frames": len(pcm), "sample_rate_hz": rate,
                       "channels": 1, "encoding": "PCM_16",
                       "duration_seconds": len(pcm) / rate}
    if np.any(encoded["segment_silenced"][a:b]):
        raise ValueError("Silenced region is not exactly zero")
    outside = np.ones(len(speech), dtype=bool)
    outside[a:b] = False
    if not np.array_equal(encoded["reference"][outside], encoded["segment_silenced"][outside]):
        raise ValueError("Silencing changed samples outside the selected interval")
    ca, cb = audit["event_samples_half_open"]
    quiet = np.ones(len(speech), dtype=bool)
    quiet[ca:cb] = False
    if not np.array_equal(encoded["reference"][quiet], encoded["machinery"][quiet]):
        raise ValueError("Mixing changed samples outside the event")
    ref = encoded["reference"].astype(np.float64) / 32768
    delta = (encoded["machinery"].astype(np.float64) - encoded["reference"].astype(np.float64)) / 32768
    effective_snr = float(10 * np.log10(np.mean(ref[a:b] ** 2) / np.mean(delta[a:b] ** 2)))
    if abs(effective_snr + 10) > 0.01:
        raise ValueError("Exported local SNR differs materially from the requested SNR")
    manifest = {
        "purpose": "New companion audio; no associated new model responses",
        "source_speech_sha256": sha(speech_path), "source_noise_sha256": sha(noise_path),
        "estimated_word_samples_half_open": [a, b], "boundary_source": "Historical automatic forced alignment",
        "human_phonetic_verification": False, "residual_cues_audited": False,
        "common_gain": gain, "common_gain_db": float(20 * np.log10(gain)),
        "quantization": "Shared-gain float32 arrays, rounded once to PCM16; no dithering or separate normalization",
        "requested_local_snr_db": -10.0,
        "effective_local_snr_after_pcm16_db": effective_snr,
        "mixture_audit_before_quantization": audit,
        "spillover_note": "Full machinery event also overlaps minutes; this is not numeral-only noise",
        "files": files,
        "verification": {"all_same_shape": True, "silenced_region_zero": True,
                         "outside_silence_matches_reference": True,
                         "outside_event_matches_reference": True,
                         "clipping_avoided": True, "model_calls": 0, "gpu_allocations": 0,
                         "browser_playback_checked": False, "human_listening_performed": False},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    manifest_path = output / "audio_manifest.json"
    serialized = json.dumps(manifest, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        raise ValueError("Existing manifest differs; use a fresh output directory to preserve its receipt")
    if not manifest_path.exists():
        manifest_path.write_text(serialized)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced_audio")
    args = parser.parse_args()
    result = prepare(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "files": result["files"],
                      "common_gain": result["common_gain"],
                      "effective_local_snr_db": result["effective_local_snr_after_pcm16_db"]}, indent=2))
