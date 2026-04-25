"""Drive and artifact helpers for the AI Shield notebooks.

These helpers keep the notebooks focused on the machine learning story.
They are intentionally small and conservative: they load and scan existing
artifacts, but they do not delete or overwrite project files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def mount_google_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive when running in Colab."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point)
    except Exception as exc:  # pragma: no cover - only used in Colab
        print(f"Drive mount skipped or unavailable: {exc}")


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def newest_existing(paths: Iterable[str | Path]) -> Path | None:
    """Return the newest existing file from a candidate list."""
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def write_json_if_allowed(obj: object, path: str | Path, force: bool = False) -> Path:
    """Write JSON only when the file is missing or force is enabled."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"Reusing existing JSON artifact: {output_path}")
        return output_path
    output_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"Saved JSON artifact: {output_path}")
    return output_path


def scan_artifacts(root: str | Path, patterns: Iterable[str]) -> dict[str, list[str]]:
    """Find matching artifact files under a root directory."""
    root_path = Path(root)
    summary: dict[str, list[str]] = {}
    for pattern in patterns:
        summary[pattern] = sorted(str(path) for path in root_path.glob(pattern))
    return summary
