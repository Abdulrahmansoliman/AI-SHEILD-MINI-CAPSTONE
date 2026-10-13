"""Read recorded evidence from original files without making an AI prediction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


MAX_FILE_BYTES = 25 * 1024 * 1024
BASE_TAGS = (
    "-FileType", "-MIMEType", "-ImageWidth", "-ImageHeight", "-EXIF:all",
    "-XMP:all", "-IPTC:all", "-PNG:all", "-QuickTime:all",
    "-FileModifyDate", "-FileCreateDate", "-Warning", "-Error",
)
CAMERA_TAGS = {"Make", "Model", "LensMake", "LensModel", "CameraModelName"}
DEVICE_TAGS = {"DeviceManufacturer", "DeviceModelName"}
SOFTWARE_TAGS = {"Software", "CreatorTool", "ProcessingSoftware"}
DATE_TAGS = {
    "DateTimeOriginal", "CreateDate", "ModifyDate", "DateCreated",
    "TimeCreated", "DigitalCreationDate", "DigitalCreationTime",
    "OffsetTime", "OffsetTimeOriginal", "OffsetTimeDigitized",
    "SubSecTime", "SubSecTimeOriginal", "SubSecTimeDigitized",
    "MetadataDate", "CreationTime",
}
FILESYSTEM_DATE_TAGS = {"FileModifyDate", "FileCreateDate"}


def _tag_name(key: str) -> str:
    return key.rsplit(":", 1)[-1]


def _select(tags: dict, names: set[str]) -> dict:
    return {key: value for key, value in tags.items() if _tag_name(key) in names}


def _error(status: str, exc: Exception) -> dict:
    return {"status": status, "error": {"type": type(exc).__name__, "message": str(exc)}}


def summarize_metadata(tags: dict[str, Any], *, include_gps: bool = False) -> dict:
    """Preserve tag groups and conflicting values; embedded claims are unverified."""
    tags = {key: value for key, value in tags.items() if key != "SourceFile"}
    gps = {key: value for key, value in tags.items() if "GPS" in key.upper()}
    if not include_gps:
        tags = {key: value for key, value in tags.items() if key not in gps}
    embedded = {
        key: value for key, value in tags.items()
        if key.split(":", 1)[0] not in {"File", "System", "ExifTool", "Composite"}
    }
    return {
        "status": "present" if embedded else "absent",
        "authenticity": "unverified",
        "camera": _select(tags, CAMERA_TAGS),
        "device": _select(tags, DEVICE_TAGS),
        "software": _select(tags, SOFTWARE_TAGS),
        "embedded_dates": _select(embedded, DATE_TAGS),
        "filesystem_dates": _select(tags, FILESYSTEM_DATE_TAGS),
        "gps": {"status": "present" if gps else "absent", "fields": gps}
        if include_gps else {"status": "not_requested", "fields": {}},
        "file_properties": _select(tags, {"FileType", "MIMEType", "ImageWidth", "ImageHeight"}),
        "warnings": _select(tags, {"Warning"}),
        "raw_tags": tags,
    }


def extract_metadata(
    path: Path, *, exiftool: str | None = None, include_gps: bool = False,
    timeout_seconds: float = 20,
) -> dict:
    executable = exiftool or os.environ.get("EXIFTOOL_PATH") or shutil.which("exiftool")
    if not executable:
        return _error("unavailable", FileNotFoundError("Install ExifTool or set EXIFTOOL_PATH."))
    try:
        result = subprocess.run(
            [executable, "-j", "-G1:4", "-a", "-s", "-n", *BASE_TAGS, str(path.resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_seconds, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ExifTool failed to read the file.")
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("Unexpected ExifTool JSON response.")
        errors = _select(payload[0], {"Error"})
        if errors:
            raise ValueError(str(errors))
        return summarize_metadata(payload[0], include_gps=include_gps)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        return _error("error", exc)


def _validation_entries(store: dict) -> list[dict]:
    results = store.get("validation_results") or {}
    entries = []
    active = results.get("activeManifest") or {}
    for severity in ("success", "informational", "failure"):
        for entry in active.get(severity) or []:
            entries.append({**entry, "severity": severity, "scope": "active_manifest"})
    for ingredient in results.get("ingredientDeltas") or []:
        delta = ingredient.get("validationDeltas") or {}
        for severity in ("success", "informational", "failure"):
            for entry in delta.get(severity) or []:
                entries.append({**entry, "severity": severity, "scope": "ingredient"})
    for entry in store.get("validation_status") or []:
        entries.append({**entry, "severity": "reported", "scope": "legacy"})
    return entries


def _check(entries: list[dict], successes: set[str], failures: set[str]) -> str:
    active = [entry for entry in entries if entry["scope"] == "active_manifest"]
    if any(entry.get("code") in failures for entry in active):
        return "failed"
    if any(entry.get("code") in successes and entry["severity"] == "success" for entry in active):
        return "verified"
    return "unknown"


def summarize_c2pa(store: dict[str, Any]) -> dict:
    """Summarize SDK validation; claims do not establish the truth of a scene."""
    label = store.get("active_manifest")
    manifest = (store.get("manifests") or {}).get(label)
    if not isinstance(manifest, dict):
        raise ValueError("C2PA response does not contain its active manifest.")
    state = store.get("validation_state", "Unknown")
    entries = _validation_entries(store)
    trust_failures = {
        "signingCredential.untrusted", "signingCredential.invalid",
        "signingCredential.revoked", "signingCredential.expired",
    }
    if state == "Trusted":
        trust = "trusted"
    elif any(entry.get("code") in trust_failures and entry["scope"] == "active_manifest"
             for entry in entries):
        trust = "not_trusted"
    else:
        trust = "not_established"
    failed = state == "Invalid" or any(
        entry["severity"] == "failure" and entry.get("code") != "signingCredential.untrusted"
        for entry in entries
    )
    if failed:
        status = "failed"
    elif state in {"Valid", "Trusted"}:
        status = "untrusted" if trust == "not_trusted" else "validated"
    else:
        status = "incomplete"
    actions = []
    source_types = []
    for assertion in manifest.get("assertions") or []:
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            for action in (assertion.get("data") or {}).get("actions") or []:
                actions.append(action)
                if action.get("digitalSourceType"):
                    source_types.append(action["digitalSourceType"])
    return {
        "status": status,
        "sdk_validation_state": state,
        "signature": _check(entries, {"claimSignature.validated"}, {"claimSignature.mismatch"}),
        "image_integrity": _check(entries, {
            "assertion.dataHash.match", "assertion.bmffHash.match", "assertion.boxesHash.match",
        }, {
            "assertion.dataHash.mismatch", "assertion.bmffHash.mismatch", "assertion.boxesHash.mismatch",
        }),
        "signer_trust": trust,
        "active_manifest": label,
        "manifest_count": len(store.get("manifests") or {}),
        "claims_binding_verified": status in {"validated", "untrusted"},
        "claims": {
            "title": manifest.get("title"),
            "claim_generator": manifest.get("claim_generator"),
            "claim_generator_info": manifest.get("claim_generator_info"),
            "signature_info": manifest.get("signature_info") or {},
            "actions": actions,
            "digital_source_types": list(dict.fromkeys(source_types)),
            "ingredients": manifest.get("ingredients") or [],
        },
        "validation_entries": entries,
        "raw_manifest_store": store,
    }


def verify_c2pa(path: Path, *, trust_anchors_pem: str | None = None) -> dict:
    policy = {
        "remote_manifest_fetch": False,
        "online_revocation_check": False,
        "trust_store": "custom_anchors" if trust_anchors_pem else "sdk_default",
    }
    try:
        import c2pa
    except (ImportError, OSError, RuntimeError) as exc:
        return {**_error("unavailable", exc), "verification_policy": policy}
    try:
        config = {"verify": {
            "verify_after_reading": True, "verify_trust": True,
            "verify_timestamp_trust": True, "remote_manifest_fetch": False, "ocsp_fetch": False,
        }}
        if trust_anchors_pem:
            config["trust"] = {"trust_anchors": trust_anchors_pem}
        with c2pa.Context.from_dict(config) as context:
            with c2pa.Reader(str(path.resolve()), context=context) as reader:
                report = summarize_c2pa(json.loads(reader.json()))
        report["sdk_version"] = importlib.metadata.version("c2pa-python")
    except c2pa.C2paError.ManifestNotFound:
        report = {"status": "absent", "signer_trust": "unknown"}
    except c2pa.C2paError.RemoteManifest as exc:
        report = _error("remote_reference", exc)
    except c2pa.C2paError.NotSupported as exc:
        report = _error("unsupported", exc)
    except (c2pa.C2paError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report = _error("error", exc)
    return {**report, "verification_policy": policy}


def analyze_provenance(
    image_path: str | Path, *, exiftool: str | None = None, include_gps: bool = False,
    trust_anchors_pem: str | None = None, max_file_bytes: int = MAX_FILE_BYTES,
) -> dict:
    """Inspect an original file. A failed branch does not suppress the other one."""
    path = Path(image_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Expected a single original image file.")
    size = path.stat().st_size
    if size == 0 or size > max_file_bytes:
        raise ValueError(f"File must contain 1 to {max_file_bytes} bytes.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = extract_metadata(path, exiftool=exiftool, include_gps=include_gps)
    provenance = verify_c2pa(path, trust_anchors_pem=trust_anchors_pem)
    remote_refs = _select(metadata.get("raw_tags", {}), {"Provenance"})
    if remote_refs and provenance["status"] == "absent":
        provenance["status"] = "remote_reference"
    if remote_refs:
        provenance["remote_references"] = remote_refs
    return {
        "schema_version": "1.0",
        "file": {"name": path.name, "size_bytes": size, "sha256": digest.hexdigest()},
        "metadata": metadata,
        "provenance": provenance,
        "interpretation": {
            "origin": "unknown",
            "missing_metadata_is_ai_evidence": False,
            "software_tags_are_ai_evidence": False,
            "detection_score_modified": False,
            "notes": [
                "Metadata fields are recorded claims and may be altered.",
                "C2PA verification checks signed claims and asset binding, not scene truth.",
                "Offline verification does not retrieve remote manifests or current revocation responses.",
                "Filesystem dates describe this file copy, not necessarily original creation.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--exiftool", help="Path to the ExifTool executable")
    parser.add_argument("--include-gps", action="store_true")
    parser.add_argument("--trust-anchors", type=Path, help="PEM trust anchors; replaces SDK defaults")
    parser.add_argument("--output", type=Path, help="Save JSON; otherwise print it")
    args = parser.parse_args()
    try:
        anchors = args.trust_anchors.read_text(encoding="utf-8") if args.trust_anchors else None
        report = analyze_provenance(
            args.image, exiftool=args.exiftool, include_gps=args.include_gps,
            trust_anchors_pem=anchors,
        )
        serialized = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
        if args.output:
            output = args.output.resolve()
            if output == args.image.resolve() or (args.trust_anchors and output == args.trust_anchors.resolve()):
                raise ValueError("Report output must not overwrite the image or trust anchors.")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="")
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
