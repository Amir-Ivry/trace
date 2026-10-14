"""Optional explicit GPU reproduction of one fixed input using the unchanged CLI.

Never called by site builds or CPU examples. Requires the separately installed
pinned runtime, snapshot, and authorized RTX PRO 6000 allocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("reference", "machinery", "segment_silenced"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/qwen-gallery")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("GPU reproduction requires explicit --allow-gpu and a permitted allocation; there is no CPU fallback")
    frozen = json.loads((ROOT / "evidence/gallery/run_manifest.json").read_text())
    request = next(r for r in frozen["inputs_in_order"] if r["condition_id"] == args.condition)
    wav = ROOT / request["input"]
    if hashlib.sha256(wav.read_bytes()).hexdigest() != request["input_sha256"]:
        parser.error("Frozen audio hash mismatch")
    source = ROOT / "source/TRACE_COMPANION_TRANSFER"
    cache = args.cache_dir.resolve()
    output = args.output.resolve()
    if source == cache or source in cache.parents or source == output or source in output.parents:
        parser.error("Cache and new outputs must stay outside the immutable transfer")
    if output.exists():
        parser.error("Output already exists; complete evidence is never overwritten")
    command = [sys.executable, "-B", str(source / "code/infer_qwen.py"),
               "--wav", str(wav), "--output", str(output), "--seed", "1234",
               "--cache-dir", str(cache), "--allow-gpu"]
    if args.allow_download:
        command.append("--allow-download")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", HF_HOME=str(ROOT / ".cache/huggingface"),
               HF_XET_CACHE=str(ROOT / ".cache/huggingface/xet"), XDG_CACHE_HOME=str(ROOT / ".cache"))
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
