from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile


INCLUDE_ITEMS = [
    "src",
    "configs",
    "scripts",
    "reports",
    "config.yaml",
    "train.py",
    "eval.py",
    "inspect_data.py",
    "validate_data.py",
    "zip_kaggle.py",
    "requirements.txt",
    "requirements-cu121.txt",
    "README.md",
]

EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def iter_files(item: Path) -> list[Path]:
    if item.is_file():
        return [item] if should_include(item) else []
    if not item.is_dir():
        return []
    files: list[Path] = []
    for root, _, names in os.walk(item):
        for name in names:
            path = Path(root) / name
            if should_include(path):
                files.append(path)
    return sorted(files)


def create_zip(output: str = "kaggle_upload.zip") -> None:
    out_path = Path(output)
    if out_path.exists():
        out_path.unlink()

    added = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_name in INCLUDE_ITEMS:
            item = Path(item_name)
            if not item.exists():
                print(f"Skipping missing item: {item_name}")
                continue
            for path in iter_files(item):
                arcname = path.as_posix()
                zf.write(path, arcname)
                added += 1
                print(f"Added {arcname}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated {out_path} with {added} files ({size_mb:.2f} MB).")
    print("Excluded by design: data/, outputs/, .venv/, .git/, checkpoints, local downloaded zips.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Kaggle-ready source zip")
    parser.add_argument("--output", default="kaggle_upload.zip", help="Output zip path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_zip(args.output)
