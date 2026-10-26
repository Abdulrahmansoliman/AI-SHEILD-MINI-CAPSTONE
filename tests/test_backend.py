import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.detector import CFDetector, DEFAULT_THRESHOLD, load_verified_config, sha256_file
from backend.pipeline import AnalysisPipeline, PipelineLimits


def image_bytes(image_format="JPEG", size=(64, 48)):
    stream = io.BytesIO()
    Image.new("RGB", size, "green").save(stream, format=image_format)
    return stream.getvalue()


class StubDetector:
    identity = {"detector_id": "test", "threshold": DEFAULT_THRESHOLD}

    def __init__(self):
        self.paths = []
        self.contents = []

    def predict(self, path):
        self.paths.append(path)
        self.contents.append(path.read_bytes())
        return {"ai_score": 0.94, "class_index": 1, "threshold": DEFAULT_THRESHOLD,
                "label": "likely_ai_generated"}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.detector = StubDetector()
        self.evidence_bytes = []
        self.gps_options = []
        self.root = Path(self.temp.name)

        def evidence(path, **options):
            self.evidence_bytes.append(path.read_bytes())
            self.gps_options.append(options["include_gps"])
            return {
                "file": {"name": path.name},
                "metadata": {"status": "absent", "raw_tags": {"SourceFile": "private"}},
                "provenance": {"status": "absent", "raw_manifest_store": {"private": True}},
                "interpretation": {"origin": "unknown", "detection_score_modified": False},
            }

        self.limits = PipelineLimits(max_bytes=4096, max_pixels=4096, temp_dir=self.root)
        self.pipeline = AnalysisPipeline(self.detector, limits=self.limits, evidence_reader=evidence)
        self.client = self.enterContext(TestClient(create_app(lambda: self.pipeline, limits=self.limits)))

    def test_both_branches_receive_original_bytes_and_cleanup(self):
        data = image_bytes()
        response = self.client.post("/analyze", files={"file": ("../photo.jpg", data, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(self.detector.contents, [data])
        self.assertEqual(self.evidence_bytes, [data])
        self.assertEqual(report["detection"]["ai_score"], 0.94)
        self.assertEqual(report["evidence"]["file"]["name"], "photo.jpg")
        self.assertFalse(report["evidence"]["interpretation"]["detection_score_modified"])
        self.assertNotIn("raw_tags", report["evidence"]["metadata"])
        self.assertNotIn("raw_manifest_store", report["evidence"]["provenance"])
        self.assertFalse(self.detector.paths[0].exists())
        self.assertFalse(list(self.root.iterdir()))

    def test_image_format_is_decoded_not_taken_from_filename(self):
        response = self.client.post("/analyze", files={"file": ("not-an-image.exe", image_bytes("PNG"), "application/octet-stream")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.detector.paths[0].suffix, ".png")

    def test_gps_defaults_off_and_requires_explicit_option(self):
        for query in ("", "?include_gps=true"):
            self.assertEqual(self.client.post("/analyze" + query, files={"file": ("photo.jpg", image_bytes())}).status_code, 200)
        self.assertEqual(self.gps_options, [False, True])

    def test_provenance_failure_does_not_suppress_detection(self):
        with patch.object(self.pipeline, "evidence_reader", side_effect=RuntimeError("internal private path")), \
             self.assertLogs("backend.pipeline", level="ERROR"):
            response = self.client.post("/analyze", files={"file": ("photo.jpg", image_bytes())})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evidence"]["provenance"]["status"], "error")
        self.assertEqual(response.json()["detection"]["ai_score"], 0.94)
        self.assertNotIn("internal private path", response.text)

    def test_invalid_images_return_422_and_cleanup(self):
        for data in (b"", b"not an image"):
            response = self.client.post("/analyze", files={"file": ("photo.jpg", data)})
            self.assertEqual(response.status_code, 422)
        self.assertFalse(list(self.root.iterdir()))
        self.assertFalse(self.detector.paths)

    def test_unsupported_image_format(self):
        response = self.client.post("/analyze", files={"file": ("photo.bmp", image_bytes("BMP", size=(16, 16)))})
        self.assertEqual(response.status_code, 422)

    def test_pixel_limit(self):
        response = self.client.post("/analyze", files={"file": ("photo.jpg", image_bytes(size=(65, 65)))})
        self.assertEqual(response.status_code, 413)
        self.assertFalse(list(self.root.iterdir()))

    def test_file_limit(self):
        response = self.client.post("/analyze", files={"file": ("photo.jpg", b"x" * 4097)})
        self.assertEqual(response.status_code, 413)
        self.assertFalse(list(self.root.iterdir()))

    def test_request_body_limit_with_and_without_content_length(self):
        data = b"x" * (4096 + 64 * 1024 + 1)
        for body in (data, iter([data[:2000], data[2000:]])):
            response = self.client.post("/analyze", content=body, headers={
                "Content-Type": "multipart/form-data; boundary=test-boundary",
            })
            self.assertEqual(response.status_code, 413)
        self.assertFalse(self.detector.paths)

    def test_busy_pipeline_returns_retryable_error(self):
        self.pipeline._busy.acquire()
        try:
            response = self.client.post("/analyze", files={"file": ("photo.jpg", image_bytes())})
        finally:
            self.pipeline._busy.release()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "2")

    def test_detection_failure_is_not_a_fabricated_prediction(self):
        with patch.object(self.detector, "predict", side_effect=RuntimeError("private model path")), \
             self.assertLogs("backend.app", level="ERROR"):
            response = self.client.post("/analyze", files={"file": ("photo.jpg", image_bytes())})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private model path", response.text)
        self.assertFalse(list(self.root.iterdir()))
        self.assertFalse(self.pipeline._busy.locked())

    def test_missing_upload_and_health(self):
        self.assertEqual(self.client.post("/analyze").status_code, 422)
        self.assertEqual(self.client.get("/health").json()["status"], "ready")

    def test_cors_does_not_accept_unconfigured_origin(self):
        response = self.client.options("/analyze", headers={
            "Origin": "https://unconfigured.example", "Access-Control-Request-Method": "POST",
        })
        self.assertNotIn("access-control-allow-origin", response.headers)


@unittest.skipUnless(os.environ.get("AI_SHIELD_CF_INTEGRATION") == "1", "Enable real checkpoint integration explicitly")
class FrozenCFIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(os.environ.get("AI_SHIELD_ARTIFACT_DIR", ".local/backend/artifacts"))
        cls.fixtures = Path(os.environ.get("AI_SHIELD_CF_FIXTURES", ".local/backend/fixtures"))
        cls.detector = CFDetector(cls.artifacts)

    def test_cpu_predictions_match_all_six_freeze_fixtures(self):
        config = self.detector.config
        policy = config["comparison"]
        for fixture in config["fixtures"]:
            with self.subTest(fixture=fixture["sample_id"]):
                path = self.fixtures / Path(fixture["path"]).name
                self.assertEqual(sha256_file(path), fixture["sha256"])
                output = self.detector.predict(path)
                self.assertTrue(math.isclose(output["ai_score"], fixture["expected_prob_fake"], rel_tol=policy["prob_rtol"], abs_tol=policy["prob_atol"]))
                self.assertTrue(math.isclose(output["logit"], fixture["expected_logit"], rel_tol=policy["logit_rtol"], abs_tol=policy["logit_atol"]))
                self.assertEqual(output["class_index"], int(fixture["expected_prob_fake"] >= DEFAULT_THRESHOLD))

    def test_real_api_combines_cf_and_provenance(self):
        path = self.fixtures / Path(self.detector.config["fixtures"][0]["path"]).name
        pipeline = AnalysisPipeline(self.detector, exiftool=os.environ.get("EXIFTOOL_PATH"))
        with TestClient(create_app(lambda: pipeline)) as client:
            response = client.post("/analyze", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["evidence"]["file"]["sha256"], sha256_file(path))
        self.assertEqual(result["detection"]["threshold"], DEFAULT_THRESHOLD)
        self.assertIn(result["evidence"]["metadata"]["status"], {"present", "absent"})
        self.assertIn(result["evidence"]["provenance"]["status"], {"validated", "untrusted", "absent", "remote_reference"})

    def test_changed_threshold_and_verification_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)
            config = json.loads((self.artifacts / "detector_config.json").read_text())
            config["default_threshold"] = 0.5
            (target / "detector_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "baseline"):
                load_verified_config(target)
            config["default_threshold"] = DEFAULT_THRESHOLD
            (target / "detector_config.json").write_text(json.dumps(config))
            (target / "verification_report.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "report hash"):
                load_verified_config(target)

    def test_wrong_checkpoint_fails_before_deserialization(self):
        from src.ai_shield_cf_inference import FrozenCF
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "best.pt"
            checkpoint.write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                FrozenCF(checkpoint, "0" * 64, DEFAULT_THRESHOLD, "0" * 64)


if __name__ == "__main__":
    unittest.main()
