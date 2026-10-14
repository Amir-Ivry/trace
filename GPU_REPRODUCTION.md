# Optional GPU reproduction

The page and the two CPU examples do not require these dependencies. This is a separate optional entry point for a permitted GPU allocation. It reuses the unchanged transferred `code/infer_qwen.py`, with provenance in `source/TRACE_COMPANION_TRANSFER/configs/code_provenance.json`.

Use Python **3.10.20** in a project-local environment. Install the exact pinned packages, without altering an existing CPU environment:

```bash
python3.10 -m venv .venv-gallery
.venv-gallery/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
.venv-gallery/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-gpu.txt
.venv-gallery/bin/python -m pip check
```

Model and processor are `Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision `26291f793822fb6be9555850f06dfe95f2d7e695`. Set `HF_HOME` and any Hugging Face/Xet caches inside this extracted project. Obtain that exact snapshot in `.cache/qwen-gallery` before holding a GPU where practical. Weights, environments, credentials, and scheduler receipts are not included in the download. Do not use another project's environments or caches.

The preserved runtime contract requires exactly one visible **NVIDIA RTX PRO 6000** with at least **94 GiB** reported GPU memory, torch **2.8.0+cu128**, transformers **5.19.0**, and matching installed implementation hashes. Use a compatible CUDA 12.8 driver. BF16, SDPA, batch size one, Thinker audio input and text/tool-call output are fixed. There is no CPU/offload, quantization, alternate hardware, or alternate-backend fallback. A failed contract check is a reason to stop and inspect the environment, not bypass the check.

Within an authorized allocation, choose a new output path and run one condition:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-gallery/bin/python -B scripts/reproduce_gpu.py reference --output outputs/reproduction/reference.json --cache-dir .cache/qwen-gallery --allow-gpu
```

The other fixed condition IDs are `machinery` and `segment_silenced`. Add `--allow-download` only if that pinned snapshot is absent. The wrapper checks the WAV hash, uses the unchanged transfer CLI, supplies the cache outside `source/`, and refuses existing outputs. It does not submit a scheduler job. No real timer action is performed.

Each request starts a fresh conversation and resets seed **1234**. The complete resolved configuration is in `evidence/gallery/run_manifest.json`: sampling enabled, temperature 0.6, top-p 0.95, top-k 20, and maximum 256 new tokens. Do not change other resolved settings. The model receives only the generic native task/tool prompt and decoded waveform, with the neutral `input.wav` placeholder. The transcript, desired duration, condition ID, alignment, source filename, archived responses, and scripted-scoring metadata are not model inputs. Do not normalize, regenerate, resample, or replace the frozen PCM16 files for these requests.

The supplied gallery milestone permits exactly one completed first response per condition, caps total GPU allocation at 60 minutes, and permits at most two additional generation attempts only after documented infrastructure failures prevented usable saved results. Outcome content, empty text, malformed calls, refusals, and token-limit truncation do not justify retrying. Release the GPU before CPU packaging or review. Later reproduction is a separate authorized run and must not overwrite the shipped observations.

Complete observations retain raw text and special tokens, token IDs, prompt, settings, revisions, seed, timestamps, termination flags, tensor hashes, and runtime. The public `raw`, `normalized`, and `validated` records preserve the distinction between generation, provenance validation, human interpretation, and audio-cue review. The corrected companion parser supplies the display interpretation; the loader's legacy parser result is provenance only. No first response is scored as a completed conversation or executed action. A stochastic reproduction need not return identical text.
