"""Independent native timer inference adapted from the validated V14 worker.

No GPU is accessed, weights loaded, or model files downloaded unless --allow-gpu
is explicitly supplied. This is a future-use entry point, not an authorized run.
The adapted CLI has not been revalidated on a GPU during transfer preparation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from native_parser import inspect_response

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(name):
    return json.loads((ROOT / "configs" / name).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_hashes(inputs, torch):
    return {
        name: {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        for name, value in inputs.items()
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, help="Any finite, mono, 16000 Hz WAV; no transcript is read.")
    parser.add_argument("--output", type=Path, help="New JSON output path. Existing files are not overwritten.")
    parser.add_argument("--seed", type=int, help="Explicit generation seed for the new project.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "model_cache", help="This project's own Hugging Face cache.")
    parser.add_argument("--allow-download", action="store_true", help="Future opt-in: obtain the pinned snapshot in --cache-dir if absent.")
    parser.add_argument("--allow-gpu", action="store_true", help="Future opt-in only after separate GPU authorization; never supplied by this package automatically.")
    parser.add_argument("--describe", action="store_true", help="Print the frozen configuration without torch, GPU, downloads, or model loading.")
    return parser.parse_args()


def main():
    args = arguments()
    spec = read_config("native_task.json")
    contract = read_config("runtime_contract.json")
    if args.describe:
        print(json.dumps({"native_task": spec, "runtime_contract": contract, "gpu_used": False, "model_loaded": False}, indent=2))
        return
    # Keep this check above all ML imports and model/cache access.
    require(args.allow_gpu, "GPU/model loading is disabled. Current transfer work authorizes no inference. A future authorized run must pass --allow-gpu explicitly; there is no CPU fallback.")
    require(args.wav is not None and args.output is not None and args.seed is not None, "Provide --wav, --output and --seed.")
    require(args.wav.is_file(), "WAV input does not exist.")
    require(not args.output.exists(), "Output exists; choose a new output filename.")
    require(0 <= args.seed < 2**32, "Seed must be in [0, 2**32).")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, GenerationConfig, set_seed
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as architecture
    import transformers.models.whisper.feature_extraction_whisper as feature

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == contract["expected_torch"], "PyTorch version differs from the validated runtime.")
    require(importlib.metadata.version("transformers") == contract["expected_transformers"], "Transformers version differs from the validated runtime.")
    require(file_hash(architecture.__file__) == contract["architecture_sha256"], "Qwen implementation hash mismatch; inspect environment, do not silently patch.")
    require(file_hash(feature.__file__) == contract["feature_extractor_sha256"], "Feature extractor hash mismatch; inspect environment, do not silently patch.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose exactly one authorized GPU; no multi-GPU or CPU fallback is implemented.")
    gpu = torch.cuda.get_device_properties(0)
    require("RTX PRO 6000" in gpu.name and gpu.total_memory >= 94 * 1024**3, "This transfer preserves the validated RTX PRO 6000 96 GB route. Other hardware requires a separately scoped qualification.")
    waveform, rate = sf.read(args.wav, dtype="float32")
    require(rate == 16000 and waveform.ndim == 1 and waveform.size > 0 and np.isfinite(waveform).all(), "Validated preprocessing requires a finite, nonempty, mono, 16000 Hz WAV. Prepare other formats separately and record the transformation.")
    wav_sha = file_hash(args.wav)
    require(float(np.max(np.abs(waveform))) > 0, "WAV is completely silent.")
    source = spec["model"]["repo"]
    revision = spec["model"]["revision"]
    load_args = dict(revision=revision, cache_dir=str(args.cache_dir.resolve()), local_files_only=not args.allow_download)
    processor = Qwen3OmniMoeProcessor.from_pretrained(source, **load_args)
    messages = [
        {"role": "system", "content": spec["native_task"]},
        {"role": "user", "content": [{"type": "audio", "audio": "input.wav"}]},
    ]
    prompt = processor.apply_chat_template(messages, tools=[spec["native_tool"]], tokenize=False, add_generation_prompt=True)
    require(digest(prompt) == contract["prompt_sha256_canonical_json_string"], "Rendered native prompt differs, including tool order. Do not continue with a changed task.")
    require(prompt == read_config("rendered_native.json")["prompt"], "Native prompt text mismatch.")
    model, loading = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        source, **load_args, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", enable_audio_output=False, output_loading_info=True,
    )
    require(not loading.get("missing_keys") and not loading.get("mismatched_keys"), "Model load has missing or mismatched parameters.")
    require(all(k.startswith(("talker.", "code2wav.")) for k in loading.get("unexpected_keys", [])), "Unexpected non-audio-output model parameters.")
    model.eval()
    model.disable_talker()
    require(all(p.device == torch.device("cuda:0") for p in model.parameters()), "Model was not placed entirely on one GPU.")
    require(all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point()), "Model parameters are not entirely BF16.")
    generation = GenerationConfig.from_dict(read_config("generation_config.json"))
    generation, unused = model.thinker._prepare_generation_config(generation)
    require(not unused, "Unexpected generation arguments.")
    require(digest(generation.to_dict()) == contract["generation_sha256_canonical_json"], "Resolved generation configuration differs from the validated configuration.")
    require(generation.eos_token_id == model.config.im_end_token_id and generation.pad_token_id == processor.tokenizer.pad_token_id, "EOS or padding token mismatch.")
    # Same processor invocation and conversion order as the successful V14 path.
    inputs = processor(text=prompt, return_tensors="pt", padding=True, audio=[waveform]).to(model.device).to(model.dtype)
    require(all(torch.is_tensor(x) for x in inputs.values()) and inputs["input_features"].numel() > 0, "Processor produced invalid audio inputs.")
    initial_hashes = tensor_hashes(inputs, torch)
    prompt_tokens = inputs["input_ids"].shape[1]
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        sequences = model.thinker.generate(**inputs, generation_config=generation)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    require(tensor_hashes(inputs, torch) == initial_hashes, "Generation modified processor inputs.")
    tokens = sequences[0, prompt_tokens:].tolist()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    eos = generation.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    result = {
        "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "historical": False, "adapted_transfer_cli": True,
        "source_model": source, "model_revision": revision, "processor_revision": revision,
        "input_wav_sha256": wav_sha, "input_sample_rate": rate, "input_samples": int(waveform.size),
        "input_tensors": initial_hashes, "messages": messages, "tools": [spec["native_tool"]],
        "rendered_prompt": prompt, "prompt_sha256_canonical_json_string": digest(prompt),
        "generation_config": generation.to_dict(), "seed": args.seed,
        "raw_response": raw, "raw_response_sha256_utf8": hashlib.sha256(raw.encode()).hexdigest(),
        "token_ids": tokens, "generated_tokens": len(tokens), "prompt_tokens": prompt_tokens,
        "stopped_at_eos": bool(tokens and tokens[-1] in eos_ids),
        "truncated": bool(len(tokens) >= generation.max_new_tokens and tokens and tokens[-1] not in eos_ids),
        "elapsed_seconds": elapsed, "proposal_inspection": inspect_response(raw),
        "model_input_contains_evaluator_metadata": False,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"), "cuda": torch.version.cuda,
            "gpu_name": gpu.name, "gpu_total_bytes": gpu.total_memory,
            "architecture_sha256": file_hash(architecture.__file__),
            "feature_extractor_sha256": file_hash(feature.__file__),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved complete response and proposal inspection to {args.output}. No real tool action was executed.")


if __name__ == "__main__":
    main()
