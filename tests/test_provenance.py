"""Regression checks for evidence interpretation and optional real-tool checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.ai_shield_provenance import (
    analyze_provenance, extract_metadata, summarize_c2pa, summarize_metadata, verify_c2pa,
)


def manifest_store(state="Valid", success=None, failure=None):
    return {
        "active_manifest": "test-manifest",
        "manifests": {"test-manifest": {"assertions": [{
            "label": "c2pa.actions.v2", "data": {"actions": [{
                "action": "c2pa.created", "digitalSourceType": "test-source-type",
            }]},
        }]}},
        "validation_state": state,
        "validation_results": {"activeManifest": {
            "success": success or [], "failure": failure or [],
        }},
    }


class EvidenceTests(unittest.TestCase):
    def test_filesystem_dates_do_not_imply_embedded_metadata(self):
        report = summarize_metadata({"SourceFile": "private/path", "System:FileModifyDate": "2026:01:01"})
        self.assertEqual(report["status"], "absent")
        self.assertFalse(report["embedded_dates"])
        self.assertTrue(report["filesystem_dates"])
        self.assertNotIn("SourceFile", report["raw_tags"])

    def test_conflicting_metadata_is_preserved(self):
        report = summarize_metadata({
            "IFD0:Make": "Canon", "IFD0:Software": "Adobe Photoshop",
            "XMP-tiff:Software": "Other editor", "ExifIFD:DateTimeOriginal": "2020:01:02 12:00:00",
        })
        self.assertEqual(len(report["software"]), 2)
        self.assertEqual(report["authenticity"], "unverified")
        self.assertTrue(report["camera"])
        self.assertTrue(report["embedded_dates"])

    def test_gps_is_opt_in_including_raw_output(self):
        tags = {"GPS:GPSLatitude": 10.5, "XMP-exif:GPSLongitude": 20.5, "IFD0:Make": "Canon"}
        report = summarize_metadata(tags)
        self.assertEqual(report["gps"]["status"], "not_requested")
        self.assertFalse(any("GPS" in key for key in report["raw_tags"]))
        self.assertEqual(summarize_metadata(tags, include_gps=True)["gps"]["status"], "present")

    def test_successes_are_required_to_report_individual_checks(self):
        report = summarize_c2pa(manifest_store())
        self.assertEqual(report["signature"], "unknown")
        self.assertEqual(report["image_integrity"], "unknown")
        self.assertEqual(report["signer_trust"], "not_established")

    def test_verified_signature_does_not_establish_signer_trust(self):
        report = summarize_c2pa(manifest_store(success=[
            {"code": "claimSignature.validated"}, {"code": "assertion.dataHash.match"},
        ]))
        self.assertEqual(report["signature"], "verified")
        self.assertEqual(report["image_integrity"], "verified")
        self.assertEqual(report["signer_trust"], "not_established")
        self.assertEqual(report["claims"]["digital_source_types"], ["test-source-type"])

    def test_trusted_manifest(self):
        self.assertEqual(summarize_c2pa(manifest_store("Trusted"))["signer_trust"], "trusted")

    def test_untrusted_signer_is_distinct_from_tampered_content(self):
        report = summarize_c2pa(manifest_store(failure=[{"code": "signingCredential.untrusted"}]))
        self.assertEqual(report["status"], "untrusted")
        self.assertEqual(report["signer_trust"], "not_trusted")

    def test_tampered_asset_keeps_signature_and_binding_separate(self):
        report = summarize_c2pa(manifest_store("Invalid", success=[
            {"code": "claimSignature.validated"},
        ], failure=[{"code": "assertion.dataHash.mismatch"}]))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["signature"], "verified")
        self.assertEqual(report["image_integrity"], "failed")

    def test_ingredient_success_is_not_active_image_verification(self):
        store = manifest_store()
        store["validation_results"]["ingredientDeltas"] = [{"validationDeltas": {
            "success": [{"code": "assertion.dataHash.match"}],
        }}]
        self.assertEqual(summarize_c2pa(store)["image_integrity"], "unknown")

    def test_missing_active_manifest_is_an_error(self):
        with self.assertRaises(ValueError):
            summarize_c2pa({"manifests": {}})

    def test_exiftool_timeout_is_not_missing_metadata(self):
        with patch("src.ai_shield_provenance.subprocess.run", side_effect=subprocess.TimeoutExpired("exiftool", 1)):
            self.assertEqual(extract_metadata(Path("image.jpg"), exiftool="tool")["status"], "error")

    def test_exiftool_corrupt_json_is_not_missing_metadata(self):
        result = subprocess.CompletedProcess([], 0, "not json", "")
        with patch("src.ai_shield_provenance.subprocess.run", return_value=result):
            self.assertEqual(extract_metadata(Path("image.jpg"), exiftool="tool")["status"], "error")

    def test_branch_errors_do_not_change_detector_or_origin(self):
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / "image.jpg"
            image.write_bytes(b"test original bytes")
            with patch("src.ai_shield_provenance.extract_metadata", return_value={"status": "error"}), \
                 patch("src.ai_shield_provenance.verify_c2pa", return_value={"status": "absent"}):
                report = analyze_provenance(image)
            self.assertEqual(report["provenance"]["status"], "absent")
            self.assertFalse(report["interpretation"]["detection_score_modified"])
            self.assertEqual(report["interpretation"]["origin"], "unknown")
            self.assertEqual(report["file"]["sha256"], hashlib.sha256(image.read_bytes()).hexdigest())

    def test_remote_reference_is_not_absent(self):
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / "image.jpg"
            image.write_bytes(b"test")
            metadata = {"status": "present", "raw_tags": {"XMP-dcterms:Provenance": "https://example.invalid/manifest"}}
            with patch("src.ai_shield_provenance.extract_metadata", return_value=metadata), \
                 patch("src.ai_shield_provenance.verify_c2pa", return_value={"status": "absent"}):
                report = analyze_provenance(image)
            self.assertEqual(report["provenance"]["status"], "remote_reference")

    def test_oversized_file_rejected_before_tools(self):
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / "image.jpg"
            image.write_bytes(b"too large")
            with self.assertRaises(ValueError):
                analyze_provenance(image, max_file_bytes=2)


@unittest.skipUnless(os.environ.get("AI_SHIELD_PROVENANCE_FIXTURES") and os.environ.get("EXIFTOOL_PATH"),
                     "Set fixture directory and EXIFTOOL_PATH for real integration tests")
class RealToolsTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = Path(os.environ["AI_SHIELD_PROVENANCE_FIXTURES"])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_camera_photo_metadata(self):
        image = self.fixtures / "P1000827.jpg"
        original = hashlib.sha256(image.read_bytes()).hexdigest()
        report = analyze_provenance(image)
        self.assertEqual(report["metadata"]["status"], "present")
        self.assertTrue(report["metadata"]["camera"])
        self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), original)

    def test_edited_photo_software_is_not_an_ai_verdict(self):
        from PIL import Image
        with Image.open(self.fixtures / "P1000827.jpg") as source:
            exif = source.getexif()
            exif[305] = "Adobe Photoshop"
            edited = self.root / "edited.jpg"
            source.convert("RGB").resize((128, 128)).save(edited, exif=exif)
        report = analyze_provenance(edited)
        self.assertIn("Adobe Photoshop", report["metadata"]["software"].values())
        self.assertFalse(report["interpretation"]["software_tags_are_ai_evidence"])

    def test_no_metadata_remains_unknown(self):
        from PIL import Image
        image = self.root / "plain.jpg"
        Image.new("RGB", (64, 64), "green").save(image)
        report = analyze_provenance(image)
        self.assertEqual(report["metadata"]["status"], "absent")
        self.assertEqual(report["provenance"]["status"], "absent")
        self.assertEqual(report["interpretation"]["origin"], "unknown")

    def test_signed_asset_has_verified_signature_and_binding(self):
        report = verify_c2pa(self.fixtures / "C.jpg")
        self.assertEqual(report["signature"], "verified")
        self.assertEqual(report["image_integrity"], "verified")
        self.assertEqual(report["status"], "untrusted")

    def test_explicit_test_anchors_establish_test_signer_trust(self):
        anchors = self.fixtures / "test-anchors.pem"
        if not anchors.is_file():
            self.skipTest("Optional upstream test trust anchors not installed")
        report = verify_c2pa(self.fixtures / "C.jpg", trust_anchors_pem=anchors.read_text())
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["signer_trust"], "trusted")
        self.assertEqual(report["verification_policy"]["trust_store"], "custom_anchors")

    def test_tampered_signed_asset_fails_image_binding(self):
        data = bytearray((self.fixtures / "C.jpg").read_bytes())
        # Change entropy-coded pixels while retaining the embedded signed manifest.
        # Walk outer JPEG segments: the manifest also contains a JPEG thumbnail.
        start = 2
        while data[start:start + 2] != b"\xff\xda":
            self.assertEqual(data[start], 255)
            segment_size = int.from_bytes(data[start + 2:start + 4], "big")
            start += 2 + segment_size
        segment_size = int.from_bytes(data[start + 2:start + 4], "big")
        scan_start = start + 2 + segment_size
        for offset in range(scan_start + 32, len(data) - 2):
            if data[offset - 1] != 255 and data[offset] not in {0, 255}:
                data[offset] ^= 1
                break
        tampered = self.root / "tampered.jpg"
        tampered.write_bytes(data)
        report = verify_c2pa(tampered)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["image_integrity"], "failed")


if __name__ == "__main__":
    unittest.main()
