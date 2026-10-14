"""Create allowlisted public CPU and static-preview packages; never publish."""
import hashlib
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TRANSFER = "source/TRACE_COMPANION_TRANSFER/"


def code_paths():
    paths = ["README.md", "CREDITS_AND_LICENSES.txt", "requirements-cpu.txt",
             "scripts/prepare_audio.py", "scripts/build_site.py", "scripts/package.py",
             "companion/PARSER_CHANGE.md", "templates/index.html", "reference/paper.pdf",
             "site/assets/style.css", "site/assets/app.js", "evidence/historical.json",
             "GPU_REPRODUCTION.md", "requirements-gpu.txt", "scripts/reproduce_gpu.py", "RESULTS_GALLERY.md",
             "scripts/prepare_github_pages.py", "PUBLISHING.md", ".gitignore"]
    for folder, pattern in [("companion", "*.py"), ("tests", "test_*.py"),
                            ("configs", "*.json"), ("prepared", "*.json"), ("prepared/audio", "*.wav")]:
        paths.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / folder).glob(pattern)))
    paths.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "evidence/gallery").rglob("*.json")))
    paths.extend(TRANSFER + item for item in [
        "code/audio_recipe.py", "configs/native_parser_fixtures.json", "configs/code_provenance.json",
        "code/infer_qwen.py", "code/native_parser.py", "configs/runtime_contract.json", "configs/rendered_native.json",
        "configs/native_prompt.txt", "configs/generation_config.json", "configs/native_task.json",
        "metadata/speech.json", "metadata/noise.json", "metadata/additive_recipe.json",
        "assets/speech/dev-real/anonymous0_24.wav", "assets/speech/CC0_1.0.txt",
        "assets/noise/95_holman-nc4_3-start1_1beat.wav", "assets/noise/CC0_1.0.txt",
        "historical/pilot_recording.wav", "historical/raw_response.txt",
        "historical/rendered_prompt.txt", "historical/generation_config.json"])
    return sorted(set(paths))


def write_zip(path, prefix, files):
    hashes = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(files):
            source = ROOT / relative
            if source.resolve() == path.resolve():
                raise ValueError("An archive cannot include itself")
            data = source.read_bytes()
            info = zipfile.ZipInfo(prefix + "/" + relative, date_time=(2026, 10, 13, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            hashes.append(hashlib.sha256(data).hexdigest() + "  " + relative)
        info = zipfile.ZipInfo(prefix + "/SHA256SUMS.txt", date_time=(2026, 10, 13, 0, 0, 0))
        archive.writestr(info, "\n".join(hashes) + "\n")


def package():
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    code = dist / "TRACE_COMPANION_CODE.zip"
    write_zip(code, "TRACE_COMPANION_CODE", code_paths())
    downloads = ROOT / "site/downloads"
    downloads.mkdir(exist_ok=True)
    shutil.copyfile(code, downloads / code.name)
    # Explicit top-level site directories avoid accidentally shipping audit or cache artifacts.
    site_paths = ["site/index.html", "site/paper.pdf"]
    for directory in ["assets", "audio", "data", "licenses", "downloads"]:
        site_paths.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "site" / directory).rglob("*")) if p.is_file())
    preview = dist / "TRACE_COMPANION_PREVIEW.zip"
    write_zip(preview, "TRACE_COMPANION_PREVIEW", site_paths)
    (dist / "SHA256SUMS.txt").write_text("".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n" for p in [preview, code]))
    print("\n".join(f"{p}: {p.stat().st_size:,} bytes" for p in [preview, code]))


if __name__ == "__main__":
    package()
