"""Load the verified CF export; do not execute downloaded Python files."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


CHECKPOINT_SHA256 = "51532d28870706a3c380a29889f634c556583bd62ed74775e5e610d5fd9d7b5f"
INFERENCE_SHA256 = "12b03eeade84fe7027ad70cc2782fb85dd857ec2e681e764b8b936f9b3725c52"
MANIFEST_SHA256 = "2ba189a3be70e6840b6be6b530253925e87d7782c2882f0189ce8e582f20d01d"
DEFAULT_THRESHOLD = 0.6339887976646423
DETECTOR_ID = "cf_finetune_v1_max_f1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_config(artifact_dir: Path) -> dict:
    config = json.loads((artifact_dir / "detector_config.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "detector_id": DETECTOR_ID,
        "architecture": "vit_small_patch16_384.augreg_in21k_ft_in1k",
        "default_threshold": DEFAULT_THRESHOLD, "default_operating_point": "validation_max_f1",
        "labels": {"0": "real", "1": "generated"}, "head_outputs": 1,
        "score": "sigmoid(logit)", "decision_operator": ">=",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Detector configuration differs from the chosen frozen baseline.")
    if (config["checkpoint"]["sha256"] != CHECKPOINT_SHA256
            or config["inference_module"]["sha256"] != INFERENCE_SHA256
            or config["source_sha256"]["experiment_manifest"] != MANIFEST_SHA256):
        raise RuntimeError("Frozen artifact identity differs.")
    if config["base_model"]["model_id"] != "OwensLab/commfor-model-384" or config["base_model"]["revision"] != "6076002bf0d9dd37537f965ee2f06f826c333b61":
        raise RuntimeError("Base model identity differs.")
    preprocess = {
        "frame": 0, "exif_transpose": True, "color_mode": "RGB", "resize_shorter_side": 440,
        "interpolation": "bilinear", "antialias": True, "center_crop": [384, 384],
        "tensor_scale": "uint8 / 255", "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225], "dtype": "float32", "tf32": False,
    }
    if config.get("preprocessing") != preprocess:
        raise RuntimeError("Frozen preprocessing differs.")
    report_path = artifact_dir / "verification_report.json"
    if sha256_file(report_path) != config["verification"]["report_sha256"]:
        raise RuntimeError("Freeze verification report hash mismatch.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (config["verification"]["status"] != "PASS" or report["status"] != "PASS"
            or report["checked"] != 6 or report["passed"] != 6
            or report["fine_tuned_checkpoint_sha256"] != CHECKPOINT_SHA256
            or report["inference_source_sha256"] != INFERENCE_SHA256):
        raise RuntimeError("Expected the passing six-image freeze verification.")
    source = Path(__file__).resolve().parents[1] / "src/ai_shield_cf_inference.py"
    if sha256_file(source) != INFERENCE_SHA256:
        raise RuntimeError("Local CF inference source differs from the verified export.")
    return config


class CFDetector:
    def __init__(self, artifact_dir: Path, *, device: str = "cpu", threads: int = 2):
        self.config = load_verified_config(artifact_dir)
        versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "timm", "Pillow")}
        for name, version in versions.items():
            if version.split("+", 1)[0] != self.config["training_libraries"][name].split("+", 1)[0]:
                raise RuntimeError(f"{name} version differs from the frozen runtime.")
        from src.ai_shield_cf_inference import FrozenCF
        import torch

        torch.set_num_threads(threads)
        self.model = FrozenCF(
            artifact_dir / "best.pt", CHECKPOINT_SHA256, DEFAULT_THRESHOLD, MANIFEST_SHA256,
            device=device,
        )
        self.identity = {
            "detector_id": DETECTOR_ID, "checkpoint_sha256": CHECKPOINT_SHA256,
            "threshold": DEFAULT_THRESHOLD, "operating_point": "validation_max_f1",
            "device": str(self.model.device), "runtime_versions": versions,
        }

    def predict(self, path: Path) -> dict:
        prediction = self.model.predict(path)
        return {
            **self.identity, "ai_score": prediction["prob_fake"], "logit": prediction["logit"],
            "class_index": prediction["predicted_label"],
            "label": "likely_ai_generated" if prediction["predicted_label"] else "likely_not_ai_generated",
        }
