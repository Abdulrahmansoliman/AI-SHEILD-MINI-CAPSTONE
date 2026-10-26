"""Combine detection and recorded evidence for one original upload."""
from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import BinaryIO, Callable

from PIL import Image, UnidentifiedImageError

from src.ai_shield_provenance import analyze_provenance, MAX_FILE_BYTES


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class InvalidImage(ValueError):
    pass


class UploadTooLarge(ValueError):
    pass


class PipelineBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineLimits:
    max_bytes: int = MAX_FILE_BYTES
    max_pixels: int = 20_000_000
    temp_dir: Path | None = None


def public_evidence(evidence: dict, filename: str) -> dict:
    evidence["file"]["name"] = filename
    evidence["metadata"].pop("raw_tags", None)
    evidence["provenance"].pop("raw_manifest_store", None)
    for name in ("metadata", "provenance"):
        if evidence[name].get("error"):
            evidence[name]["error"] = {
                "type": evidence[name]["error"]["type"],
                "message": f"{name.capitalize()} processing could not complete.",
            }
    return evidence


class AnalysisPipeline:
    def __init__(
        self, detector, *, limits: PipelineLimits | None = None,
        evidence_reader: Callable = analyze_provenance, exiftool: str | None = None,
        trust_anchors_pem: str | None = None,
    ):
        self.detector = detector
        self.limits = limits or PipelineLimits()
        self.evidence_reader = evidence_reader
        self.exiftool = exiftool
        self.trust_anchors_pem = trust_anchors_pem
        self._busy = threading.Lock()

    def analyze(self, stream: BinaryIO, filename: str | None, *, include_gps: bool = False) -> dict:
        if not self._busy.acquire(blocking=False):
            raise PipelineBusy("An image is already being processed. Try again shortly.")
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="ai_shield_", dir=self.limits.temp_dir) as folder:
                path = Path(folder) / "original"
                size = 0
                with path.open("wb") as destination:
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        if size > self.limits.max_bytes:
                            raise UploadTooLarge("Image exceeds the upload limit.")
                        destination.write(chunk)
                if size == 0:
                    raise InvalidImage("Image is empty.")
                try:
                    with Image.open(path) as source:
                        if source.format not in IMAGE_SUFFIXES:
                            raise InvalidImage("Supported formats are JPEG, PNG, and WebP.")
                        if source.width * source.height > self.limits.max_pixels:
                            raise UploadTooLarge("Image exceeds the pixel limit.")
                        image_format = source.format
                        source.seek(0)
                        source.load()
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                    raise InvalidImage("Could not decode the uploaded image.") from exc
                # Use the decoded format rather than trusting an uploaded filename.
                path = path.rename(path.with_suffix(IMAGE_SUFFIXES[image_format]))
                evidence_started = time.perf_counter()
                try:
                    evidence = self.evidence_reader(
                        path, exiftool=self.exiftool, include_gps=include_gps,
                        trust_anchors_pem=self.trust_anchors_pem, max_file_bytes=self.limits.max_bytes,
                    )
                except Exception:
                    LOGGER.exception("Provenance processing failed")
                    evidence = {
                        "schema_version": "1.0", "file": {"size_bytes": size},
                        "metadata": {"status": "error"}, "provenance": {"status": "error"},
                        "interpretation": {"origin": "unknown", "detection_score_modified": False},
                    }
                evidence_seconds = time.perf_counter() - evidence_started
                detection_started = time.perf_counter()
                detection = self.detector.predict(path)
                detection_seconds = time.perf_counter() - detection_started
                safe_name = Path((filename or "image").replace("\\", "/")).name[:255]
                safe_name = "".join(char for char in safe_name if ord(char) >= 32) or "image"
                return {
                    "schema_version": "1.0", "detection": detection,
                    "evidence": public_evidence(evidence, safe_name),
                    "timing_seconds": {
                        "detection": round(detection_seconds, 4),
                        "evidence": round(evidence_seconds, 4),
                        "total": round(time.perf_counter() - started, 4),
                    },
                }
        finally:
            self._busy.release()
