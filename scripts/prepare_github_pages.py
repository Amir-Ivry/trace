"""Build the public code download and the docs/ GitHub Pages directory on CPU.

docs/ is generated output owned by this script. No network or deployment occurs.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for script in ("build_site.py", "package.py"):
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script)],
                       cwd=ROOT, check=True)
    destination = ROOT / "docs"
    if destination.is_symlink():
        raise ValueError("Refusing to replace a symlink at docs/")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ROOT / "site", destination)
    (destination / ".nojekyll").write_text("")
    print(f"GitHub Pages files ready: {destination}")


if __name__ == "__main__":
    main()
