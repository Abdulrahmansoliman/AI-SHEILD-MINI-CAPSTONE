"""Check notebook syntax and pass/fail reporting without ML dependencies.

Uses synthetic predictions to exercise the actual notebook reporting/export cells.
It does not claim that the real checkpoint reproduces its saved predictions.
"""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import types
import unittest
import uuid


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/AI_Shield_CF_Freeze_v1.ipynb"
DOCUMENT = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
SOURCES = ["".join(cell["source"]) for cell in DOCUMENT["cells"]
           if cell["cell_type"] == "code"]
HELPERS = next(source for source in SOURCES if "def sha256_file" in source)
COMPARE = next(source for source in SOURCES if "run_id = datetime" in source)
EXPORT = next(source for source in SOURCES if 'require(verification["status"]' in source)
MODULE_CELL = next(source for source in SOURCES if source.startswith("MODULE_SOURCE ="))
MODULE_SOURCE = ast.literal_eval(ast.parse(MODULE_CELL).body[0].value)


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_csv(self, path, index=False):
        fields = list(dict.fromkeys(key for row in self.rows for key in row))
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)


class FreezeNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        checkpoint = root / "best.pt"
        checkpoint.write_bytes(b"synthetic checkpoint for reporting tests")
        lock_path = root / "detector_lock.json"
        lock_path.write_text("{}\n", encoding="utf-8")
        fixtures = []
        for index, score in enumerate((0.1, 0.3, 0.6, 0.7, 0.85, 0.95)):
            path = root / f"fixture_{index}.bin"
            path.write_bytes(f"fixture {index}".encode())
            fixtures.append({
                "sample_id": str(index), "label": int(index >= 3), "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_prob_fake": score, "expected_logit": math.log(score / (1 - score)),
            })
        self.fixture_by_path = {fixture["path"]: fixture for fixture in fixtures}
        self.env = {
            "Path": Path, "hashlib": hashlib, "json": json,
            "datetime": datetime, "timezone": timezone, "uuid": uuid,
            "np": types.SimpleNamespace(isclose=lambda a, b, atol, rtol:
                                        abs(a - b) <= atol + rtol * abs(b)),
            "pd": types.SimpleNamespace(DataFrame=FakeFrame),
            "display": lambda value: None, "print": lambda *args: None,
            "torch": types.SimpleNamespace(
                version=types.SimpleNamespace(cuda=None),
                backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: None))),
            "FREEZE_ROOT": root, "CHECKPOINT": checkpoint, "LOCK_PATH": lock_path,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "DEFAULT_THRESHOLD": 0.6339887976646423,
            "PROB_ATOL": 1e-5, "PROB_RTOL": 1e-5,
            "LOGIT_ATOL": 1e-4, "LOGIT_RTOL": 1e-5,
            "SAMPLES_PER_CLASS": 3, "fixtures": fixtures, "lock": {"comparison": {}},
            "MODULE_SOURCE": MODULE_SOURCE, "runtime_versions": {"test": "synthetic"},
        }
        helper_tree = ast.parse(HELPERS)
        functions = ast.Module(body=[node for node in helper_tree.body
                                     if isinstance(node, ast.FunctionDef)], type_ignores=[])
        exec(compile(functions, "helpers", "exec"), self.env)
        self.env["detector"] = types.SimpleNamespace(
            device=types.SimpleNamespace(type="cpu"), predict=self.predict)

    def predict(self, path):
        fixture = self.fixture_by_path[path]
        return {"prob_fake": fixture["expected_prob_fake"],
                "logit": fixture["expected_logit"],
                "predicted_label": int(fixture["expected_prob_fake"] >= self.env["DEFAULT_THRESHOLD"])}

    def run_comparison(self):
        exec(compile(COMPARE, "comparison_cell", "exec"), self.env)

    def run_export(self):
        exec(compile(EXPORT, "export_cell", "exec"), self.env)

    def assert_failed_report(self):
        report_path = self.env["RUN_OUTPUT"] / "verification_report.json"
        self.assertEqual(json.loads(report_path.read_text())["status"], "FAIL")
        with self.assertRaises(RuntimeError):
            self.run_export()
        self.assertFalse((self.env["RUN_OUTPUT"] / "detector_config.json").exists())

    def test_notebook_and_module_compile(self):
        self.assertEqual(DOCUMENT["nbformat"], 4)
        ids = [cell["id"] for cell in DOCUMENT["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for index, source in enumerate(SOURCES):
            compile(source, f"cell_{index}", "exec")
        compile(MODULE_SOURCE, "ai_shield_cf_inference.py", "exec")

    def test_matching_predictions_export_pass(self):
        self.run_comparison()
        self.run_export()
        output = self.env["RUN_OUTPUT"]
        config = json.loads((output / "detector_config.json").read_text())
        self.assertEqual(config["verification"]["status"], "PASS")
        self.assertEqual(config["verification"]["checked"], 6)
        self.assertEqual((output / "ai_shield_cf_inference.py").read_text(), MODULE_SOURCE)

    def test_score_mismatch_saves_failure(self):
        def wrong(path):
            result = self.predict(path)
            result["prob_fake"] += 0.02
            return result
        self.env["detector"].predict = wrong
        with self.assertRaises(RuntimeError):
            self.run_comparison()
        self.assert_failed_report()

    def test_classification_mismatch_fails_even_when_scores_match(self):
        def wrong(path):
            result = self.predict(path)
            result["predicted_label"] = 1 - result["predicted_label"]
            return result
        self.env["detector"].predict = wrong
        with self.assertRaises(RuntimeError):
            self.run_comparison()
        self.assert_failed_report()

    def test_changed_image_is_not_skipped(self):
        Path(self.env["fixtures"][0]["path"]).write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            self.run_comparison()
        self.assert_failed_report()
        self.assertEqual(self.env["verification"]["passed"], 5)

    def test_changed_checkpoint_blocks_export(self):
        self.run_comparison()
        self.env["CHECKPOINT"].write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            self.run_export()

    def test_changed_module_blocks_export(self):
        self.run_comparison()
        self.env["MODULE_SOURCE"] += "\n# changed\n"
        with self.assertRaises(RuntimeError):
            self.run_export()

    def test_lock_cannot_be_overwritten(self):
        write_once = self.env["write_once"]
        write_once(self.env["LOCK_PATH"], "{}\n")
        with self.assertRaises(RuntimeError):
            write_once(self.env["LOCK_PATH"], "changed\n")
        self.assertEqual(self.env["LOCK_PATH"].read_text(), "{}\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
