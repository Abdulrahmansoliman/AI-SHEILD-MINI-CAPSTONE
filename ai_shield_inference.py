from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE_ZIP = BASE_DIR / "ai_shield_deployment_bundle.zip"
DEFAULT_ARTIFACT_DIR = BASE_DIR / "deployment_artifacts"


class AIShieldDependencyError(RuntimeError):
    """Raised when an optional ML dependency is missing at inference time."""


def ensure_artifacts(
    bundle_zip: Path = DEFAULT_BUNDLE_ZIP,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Path:
    """Return an artifact directory, extracting the bundle once if needed."""
    artifact_dir = Path(artifact_dir)
    models_dir = artifact_dir / "models"
    schema_path = artifact_dir / "feature_schema.json"
    if models_dir.exists() and schema_path.exists():
        return artifact_dir

    if not bundle_zip.exists():
        raise FileNotFoundError(
            "Could not find deployment artifacts. Put ai_shield_deployment_bundle.zip "
            f"next to the app or extract it into {artifact_dir}."
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip, "r") as archive:
        for member in archive.infolist():
            target = (artifact_dir / member.filename).resolve()
            if not str(target).startswith(str(artifact_dir.resolve())):
                raise RuntimeError(f"Unsafe path inside artifact bundle: {member.filename}")
        archive.extractall(artifact_dir)
    return artifact_dir


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class AIShieldPaths:
    artifact_dir: Path
    schema_path: Path
    manifest_path: Path
    effnet_checkpoint: Path
    effnet_calibrator: Path
    vae_checkpoint: Path
    vae_reference_errors: Path
    openforensics_json: Path
    openforensics_h5: Path
    semantic_checkpoint: Path
    phase1_forensic_glm: Path
    phase3_glm: Path
    phase3_metrics: Path
    phase3_coefficients: Path

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> "AIShieldPaths":
        artifact_dir = Path(artifact_dir)
        models = artifact_dir / "models"
        return cls(
            artifact_dir=artifact_dir,
            schema_path=artifact_dir / "feature_schema.json",
            manifest_path=artifact_dir / "artifact_manifest.json",
            effnet_checkpoint=models
            / "phase1_effnet"
            / "effnet_b0_tinygenimage_full_balanced_v1_best.pt",
            effnet_calibrator=models / "phase1_effnet" / "platt_calibrator.pkl",
            vae_checkpoint=models / "phase1_vae" / "vae_real_only_tinygenimage_v1_best.pt",
            vae_reference_errors=models / "phase1_vae" / "vae_reconstruction_errors.npz",
            openforensics_json=models / "openforensics" / "dffnetv2B0.json",
            openforensics_h5=models / "openforensics" / "dffnetv2B0.h5",
            semantic_checkpoint=models
            / "phase2_semantic"
            / "semantic_vit_transformer_attention_best.pt",
            phase1_forensic_glm=models / "phase3_fusion" / "phase3_forensic_only_glm_v1.pkl",
            phase3_glm=models / "phase3_fusion" / "phase3_full_forensic_semantic_glm_v1.pkl",
            phase3_metrics=models / "phase3_fusion" / "phase3_fusion_metrics_v1.json",
            phase3_coefficients=models / "phase3_fusion" / "phase3_full_glm_coefficients_v1.csv",
        )


@dataclass
class BranchResult:
    name: str
    score: float | None
    status: str
    details: dict[str, float | int | str]


@dataclass
class PredictionResult:
    forensic_prob_fake: float
    forensic_label: str
    forensic_threshold: float
    final_prob_fake: float
    final_label: str
    threshold: float
    features: pd.DataFrame
    branches: dict[str, BranchResult]
    image_size: tuple[int, int]


def normalize_pil_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


def _safe_torch_load(torch_module, path: Path, device: str):
    try:
        return torch_module.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=device)


def _checkpoint_state(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return payload
    return payload


def _load_state_flexibly(model, state: dict) -> None:
    attempts = [state]
    if any(key.startswith("module.") for key in state):
        attempts.append({key.removeprefix("module."): value for key, value in state.items()})
    if any(key.startswith("model.") for key in state):
        attempts.append({key.removeprefix("model."): value for key, value in state.items()})
    if not any(key.startswith("model.") for key in state):
        attempts.append({f"model.{key}": value for key, value in state.items()})

    last_error: Exception | None = None
    for candidate in attempts:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except Exception as exc:
            last_error = exc
        try:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            model_keys = len(model.state_dict())
            if len(missing) < model_keys * 0.25 and len(unexpected) < len(candidate) * 0.25:
                return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load checkpoint state into model: {last_error}")


def _image_to_tensor(image: Image.Image, size: int, normalize: bool, torch_module):
    image = normalize_pil_image(image).resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
    tensor = torch_module.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


@dataclass
class SemanticConfig:
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    transformer_dim: int = 256
    transformer_num_heads: int = 4
    transformer_depth: int = 1
    transformer_mlp_ratio: float = 2.0
    attention_dim: int = 256
    classifier_hidden_dim: int = 256
    dropout: float = 0.15
    freeze_vit_backbone: bool = True


class FinalAIShieldInference:
    """Loads the frozen Phase 1/2 branches and the Phase 3 GLM meta-layer."""

    def __init__(self, paths: AIShieldPaths, device: str | None = None):
        self.paths = paths
        self.schema = read_json(paths.schema_path)
        self.threshold = float(self.schema.get("recommended_threshold_from_phase3_validation", 0.5))
        self.feature_order = list(self.schema["full_feature_order"])
        self.device = device
        self._effnet = None
        self._calibrator = None
        self._vae = None
        self._vae_reference = None
        self._openforensics = None
        self._semantic = None
        self._semantic_processor = None
        self._forensic_glm = None
        self._glm = None

    @classmethod
    def from_bundle(
        cls,
        bundle_zip: Path = DEFAULT_BUNDLE_ZIP,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
        device: str | None = None,
    ) -> "FinalAIShieldInference":
        artifact_dir = ensure_artifacts(bundle_zip=bundle_zip, artifact_dir=artifact_dir)
        return cls(AIShieldPaths.from_artifact_dir(artifact_dir), device=device)

    def _torch_device(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError(
                "PyTorch is required for the EfficientNet, VAE, and semantic branches. "
                "Install requirements.txt first."
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch, device

    def load_effnet(self):
        if self._effnet is not None:
            return self._effnet
        torch, device = self._torch_device()
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError("The timm package is required for EfficientNet-B0.") from exc
        import torch.nn as nn

        class EfficientNetBinaryClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)

            def forward(self, x):
                return self.model(x).view(-1)

        model = EfficientNetBinaryClassifier().to(device)
        payload = _safe_torch_load(torch, self.paths.effnet_checkpoint, device)
        _load_state_flexibly(model, _checkpoint_state(payload))
        model.eval()
        self._effnet = model
        return model

    def load_calibrator(self):
        if self._calibrator is not None:
            return self._calibrator
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError("joblib is required to load the Platt calibrator.") from exc
        self._calibrator = joblib.load(self.paths.effnet_calibrator)
        return self._calibrator

    def predict_effnet(self, image: Image.Image) -> BranchResult:
        torch, device = self._torch_device()
        model = self.load_effnet()
        tensor = _image_to_tensor(image, size=256, normalize=True, torch_module=torch).to(device)
        with torch.no_grad():
            logit = float(model(tensor).detach().cpu().item())
        prob = sigmoid(logit)

        calibrated = prob
        try:
            calibrator = self.load_calibrator()
            if hasattr(calibrator, "predict_proba"):
                calibrated = float(calibrator.predict_proba(np.array([[logit]], dtype=np.float32))[0, 1])
            elif hasattr(calibrator, "predict"):
                calibrated = float(np.asarray(calibrator.predict(np.array([[logit]], dtype=np.float32))).ravel()[0])
        except Exception:
            calibrated = prob

        return BranchResult(
            name="EfficientNet-B0 forensic classifier",
            score=calibrated,
            status="scored",
            details={
                "effnet_logit": logit,
                "effnet_prob": prob,
                "effnet_calibrated_prob": float(np.clip(calibrated, 0.0, 1.0)),
            },
        )

    def load_vae(self):
        if self._vae is not None:
            return self._vae
        torch, device = self._torch_device()
        import torch.nn as nn

        class ConvVAE(nn.Module):
            def __init__(self, latent_dim=128):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.ReLU(inplace=True),
                )
                self.flatten_dim = 256 * 8 * 8
                self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
                self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
                self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),
                    nn.Sigmoid(),
                )

            def encode(self, x):
                hidden = self.encoder(x).view(x.size(0), -1)
                return self.fc_mu(hidden), self.fc_logvar(hidden)

            def decode(self, z):
                hidden = self.fc_decode(z).view(z.size(0), 256, 8, 8)
                return self.decoder(hidden)

            def forward(self, x):
                mu, _ = self.encode(x)
                return self.decode(mu)

        model = ConvVAE().to(device)
        payload = _safe_torch_load(torch, self.paths.vae_checkpoint, device)
        _load_state_flexibly(model, _checkpoint_state(payload))
        model.eval()
        self._vae = model
        return model

    def load_vae_reference(self) -> np.ndarray:
        if self._vae_reference is not None:
            return self._vae_reference
        arrays = []
        with np.load(self.paths.vae_reference_errors, allow_pickle=False) as data:
            for key in data.files:
                value = np.asarray(data[key]).astype(float).ravel()
                if value.size and ("error" in key.lower() or "recon" in key.lower()):
                    arrays.append(value[np.isfinite(value)])
            if not arrays:
                for key in data.files:
                    value = np.asarray(data[key])
                    if np.issubdtype(value.dtype, np.number):
                        arrays.append(value.astype(float).ravel())
        reference = np.concatenate(arrays) if arrays else np.array([0.0, 1.0], dtype=float)
        self._vae_reference = reference[np.isfinite(reference)]
        return self._vae_reference

    def predict_vae(self, image: Image.Image) -> BranchResult:
        torch, device = self._torch_device()
        model = self.load_vae()
        tensor = _image_to_tensor(image, size=128, normalize=False, torch_module=torch).to(device)
        with torch.no_grad():
            reconstruction = model(tensor)
            error = float(torch.mean((tensor - reconstruction) ** 2).detach().cpu().item())
        reference = self.load_vae_reference()
        percentile = float(np.mean(reference <= error)) if reference.size else 0.5
        return BranchResult(
            name="Real-only VAE anomaly branch",
            score=percentile,
            status="scored",
            details={
                "vae_recon_error": error,
                "vae_recon_percentile": float(np.clip(percentile, 0.0, 1.0)),
            },
        )

    def detect_faces(self, image: Image.Image) -> BranchResult:
        try:
            import cv2
        except ModuleNotFoundError:
            return BranchResult(
                name="Face gate",
                score=None,
                status="opencv_missing",
                details={
                    "face_present": 0,
                    "num_faces": 0,
                    "largest_face_ratio": 0.0,
                    "face_confidence": 0.0,
                },
            )

        rgb = np.asarray(normalize_pil_image(image))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return BranchResult(
                name="Face gate",
                score=None,
                status="cascade_missing",
                details={
                    "face_present": 0,
                    "num_faces": 0,
                    "largest_face_ratio": 0.0,
                    "face_confidence": 0.0,
                },
            )

        image_area = max(1, rgb.shape[0] * rgb.shape[1])
        faces = []
        try:
            boxes, _reject, weights = detector.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                outputRejectLevels=True,
            )
        except Exception:
            boxes = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            weights = [1.0] * len(boxes)

        for box, weight in zip(boxes, weights):
            x, y, w, h = [int(item) for item in box]
            area_ratio = float((w * h) / image_area)
            confidence = float(np.asarray(weight).ravel()[0]) if np.asarray(weight).size else 0.0
            faces.append({"area_ratio": area_ratio, "confidence": confidence})

        valid_faces = [
            face
            for face in faces
            if face["confidence"] >= 0.0 and face["area_ratio"] >= 0.01
        ]
        largest = max((face["area_ratio"] for face in valid_faces), default=0.0)
        max_conf = max((face["confidence"] for face in valid_faces), default=0.0)
        face_present = int(len(valid_faces) > 0)
        return BranchResult(
            name="Computer vision face gate",
            score=float(face_present),
            status="face_found" if face_present else "no_valid_face",
            details={
                "face_present": face_present,
                "num_faces": int(len(valid_faces)),
                "largest_face_ratio": float(largest),
                "face_confidence": float(max_conf),
            },
        )

    def load_openforensics(self):
        if self._openforensics is not None:
            return self._openforensics
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError("TensorFlow is required for the OpenForensics face branch.") from exc

        model = None
        json_text = self.paths.openforensics_json.read_text(encoding="utf-8")
        try:
            model = tf.keras.models.model_from_json(json_text)
            model.load_weights(str(self.paths.openforensics_h5))
        except Exception:
            base = tf.keras.applications.EfficientNetV2B0(
                weights=None,
                include_top=False,
                input_shape=(256, 256, 3),
                pooling="max",
                include_preprocessing=True,
            )
            model = tf.keras.Sequential(
                [base, tf.keras.layers.Dense(1, activation="sigmoid", name="dense_2")],
                name="dffnetv2b0",
            )
            _ = model(np.zeros((1, 256, 256, 3), dtype=np.float32), training=False)
            model.load_weights(str(self.paths.openforensics_h5))
        self._openforensics = model
        return model

    def predict_openforensics(self, image: Image.Image, face_branch: BranchResult) -> BranchResult:
        if int(face_branch.details["face_present"]) == 0:
            return BranchResult(
                name="OpenForensics face detector",
                score=0.5,
                status="not_face_applicable",
                details={
                    "openforensics_applicable": 0,
                    "openforensics_prob_fake": 0.5,
                },
            )
        try:
            model = self.load_openforensics()
        except Exception:
            return BranchResult(
                name="OpenForensics face detector",
                score=0.5,
                status="model_unavailable_neutral",
                details={
                    "openforensics_applicable": 0,
                    "openforensics_prob_fake": 0.5,
                },
            )
        resized = normalize_pil_image(image).resize((256, 256))
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        real_prob = float(np.asarray(model.predict(batch, verbose=0)).ravel()[0])
        fake_prob = float(np.clip(1.0 - real_prob, 0.0, 1.0))
        return BranchResult(
            name="OpenForensics face branch",
            score=fake_prob,
            status="scored",
            details={
                "openforensics_applicable": 1,
                "openforensics_prob_fake": fake_prob,
            },
        )

    def load_semantic(self):
        if self._semantic is not None and self._semantic_processor is not None:
            return self._semantic, self._semantic_processor
        torch, device = self._torch_device()
        import torch.nn as nn
        import torch.nn.functional as F
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError(
                "transformers is required for the semantic ViT attention branch."
            ) from exc

        class CustomMultiHeadSelfAttention(nn.Module):
            def __init__(self, d_model, num_heads, dropout=0.15):
                super().__init__()
                if d_model % num_heads != 0:
                    raise ValueError("d_model must be divisible by num_heads")
                self.d_model = d_model
                self.num_heads = num_heads
                self.head_dim = d_model // num_heads
                self.scale = self.head_dim ** -0.5
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(dropout)

            def _split_heads(self, x):
                batch_size, num_tokens, _ = x.shape
                x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)
                return x.transpose(1, 2)

            def _merge_heads(self, x):
                batch_size, num_heads, num_tokens, head_dim = x.shape
                x = x.transpose(1, 2).contiguous()
                return x.view(batch_size, num_tokens, num_heads * head_dim)

            def forward(self, x):
                q = self._split_heads(self.q_proj(x))
                k = self._split_heads(self.k_proj(x))
                v = self._split_heads(self.v_proj(x))
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attention = torch.softmax(scores, dim=-1)
                attention_for_summary = attention
                attention = self.dropout(attention)
                context = torch.matmul(attention, v)
                output = self.out_proj(self._merge_heads(context))
                return output, attention_for_summary

        class SemanticTransformerBlock(nn.Module):
            def __init__(self, d_model, num_heads, mlp_ratio=2.0, dropout=0.15):
                super().__init__()
                hidden_dim = int(d_model * mlp_ratio)
                self.norm1 = nn.LayerNorm(d_model)
                self.attention = CustomMultiHeadSelfAttention(d_model, num_heads, dropout)
                self.dropout1 = nn.Dropout(dropout)
                self.norm2 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_model),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                attn_output, attention = self.attention(self.norm1(x))
                x = x + self.dropout1(attn_output)
                x = x + self.mlp(self.norm2(x))
                return x, attention

        class SemanticTransformerHead(nn.Module):
            def __init__(
                self,
                hidden_size,
                d_model=256,
                num_heads=4,
                depth=1,
                mlp_ratio=2.0,
                attention_dim=256,
                classifier_hidden_dim=256,
                dropout=0.15,
            ):
                super().__init__()
                self.input_projection = nn.Linear(hidden_size, d_model)
                self.blocks = nn.ModuleList(
                    [
                        SemanticTransformerBlock(d_model, num_heads, mlp_ratio, dropout)
                        for _ in range(depth)
                    ]
                )
                self.final_norm = nn.LayerNorm(d_model)
                self.pool_key = nn.Linear(d_model, attention_dim)
                self.pool_energy = nn.Linear(attention_dim, 1)
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(d_model, classifier_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden_dim, 1),
                )

            def _attention_summary(self, attention_maps):
                eps = 1e-8
                summaries = []
                for attention in attention_maps:
                    num_tokens = attention.shape[-1]
                    entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)
                    entropy = entropy / math.log(num_tokens)
                    focus = attention.max(dim=-1).values
                    summaries.append((entropy.mean(dim=(1, 2)), focus.mean(dim=(1, 2))))
                entropy = torch.stack([item[0] for item in summaries], dim=0).mean(dim=0)
                focus = torch.stack([item[1] for item in summaries], dim=0).mean(dim=0)
                return entropy, focus

            def forward(self, patch_tokens):
                x = self.input_projection(patch_tokens)
                self_attention_maps = []
                for block in self.blocks:
                    x, attention = block(x)
                    self_attention_maps.append(attention)
                x = self.final_norm(x)
                pool_energy = self.pool_energy(torch.tanh(self.pool_key(x))).squeeze(-1)
                pool_weights = torch.softmax(pool_energy, dim=1)
                pooled = torch.sum(pool_weights.unsqueeze(-1) * x, dim=1)
                logits = self.classifier(pooled).squeeze(-1)
                eps = 1e-8
                pool_entropy = -(pool_weights * torch.log(pool_weights + eps)).sum(dim=1)
                pool_entropy = pool_entropy / math.log(pool_weights.shape[1])
                pool_focus = pool_weights.max(dim=1).values
                self_entropy, self_focus = self._attention_summary(self_attention_maps)
                return {
                    "logits": logits,
                    "attention_weights": pool_weights,
                    "attention_entropy": pool_entropy,
                    "attention_focus": pool_focus,
                    "self_attention_entropy": self_entropy,
                    "self_attention_focus": self_focus,
                    "semantic_feature": pooled,
                }

        class ViTSemanticAttentionClassifier(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.vit = AutoModel.from_pretrained(config.vit_model_name)
                self.semantic_head = SemanticTransformerHead(
                    hidden_size=self.vit.config.hidden_size,
                    d_model=config.transformer_dim,
                    num_heads=config.transformer_num_heads,
                    depth=config.transformer_depth,
                    mlp_ratio=config.transformer_mlp_ratio,
                    attention_dim=config.attention_dim,
                    classifier_hidden_dim=config.classifier_hidden_dim,
                    dropout=config.dropout,
                )
                if config.freeze_vit_backbone:
                    for param in self.vit.parameters():
                        param.requires_grad = False

            def forward(self, pixel_values, labels=None):
                vit_outputs = self.vit(pixel_values=pixel_values, return_dict=True)
                patch_tokens = vit_outputs.last_hidden_state[:, 1:, :]
                output = self.semantic_head(patch_tokens)
                if labels is not None:
                    output["loss"] = F.binary_cross_entropy_with_logits(
                        output["logits"], labels.float()
                    )
                return output

        config = SemanticConfig()
        processor = AutoImageProcessor.from_pretrained(config.vit_model_name)
        model = ViTSemanticAttentionClassifier(config).to(device)
        payload = _safe_torch_load(torch, self.paths.semantic_checkpoint, device)
        _load_state_flexibly(model, _checkpoint_state(payload))
        model.eval()
        self._semantic = model
        self._semantic_processor = processor
        return model, processor

    def predict_semantic(self, image: Image.Image) -> BranchResult:
        torch, device = self._torch_device()
        model, processor = self.load_semantic()
        inputs = processor(images=normalize_pil_image(image), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            output = model(pixel_values)
        logit = float(output["logits"].detach().cpu().item())
        prob = sigmoid(logit)
        details = {
            "semantic_vit_logit": logit,
            "semantic_vit_prob": prob,
            "semantic_attention_entropy": float(output["attention_entropy"].detach().cpu().item()),
            "semantic_attention_focus": float(output["attention_focus"].detach().cpu().item()),
            "semantic_self_attention_entropy": float(
                output["self_attention_entropy"].detach().cpu().item()
            ),
            "semantic_self_attention_focus": float(
                output["self_attention_focus"].detach().cpu().item()
            ),
        }
        return BranchResult(
            name="Semantic ViT attention branch",
            score=prob,
            status="scored",
            details=details,
        )

    def load_glm(self):
        if self._glm is not None:
            return self._glm
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError("joblib is required to load the final GLM.") from exc
        self._glm = joblib.load(self.paths.phase3_glm)
        self._patch_sklearn_pickle_compatibility(self._glm)
        return self._glm

    def load_forensic_glm(self):
        if self._forensic_glm is not None:
            return self._forensic_glm
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise AIShieldDependencyError("joblib is required to load the Phase 1 forensic GLM.") from exc
        self._forensic_glm = joblib.load(self.paths.phase1_forensic_glm)
        self._patch_sklearn_pickle_compatibility(self._forensic_glm)
        return self._forensic_glm

    def forensic_threshold(self) -> float:
        metrics = self.load_metrics()
        threshold = (
            metrics.get("models", {})
            .get("forensic_only_glm", {})
            .get("validation", {})
            .get("threshold", 0.5)
        )
        return float(threshold)

    @staticmethod
    def _patch_sklearn_pickle_compatibility(model) -> None:
        """Patch small sklearn private-attribute drift in old saved pipelines.

        The Phase 3 GLM was saved as a sklearn Pipeline in Colab. Hugging Face may
        install a newer sklearn build, and some sklearn estimators add private
        attributes between versions. The learned public state is still valid; this
        only restores missing internal defaults needed by transform/predict_proba.
        """
        steps = []
        if hasattr(model, "steps"):
            steps.extend(step for _, step in model.steps)
        if hasattr(model, "named_steps"):
            steps.extend(model.named_steps.values())

        for step in steps:
            if step.__class__.__name__ == "SimpleImputer" and not hasattr(step, "_fill_dtype"):
                statistics = getattr(step, "statistics_", None)
                if statistics is not None:
                    step._fill_dtype = np.asarray(statistics).dtype
                else:
                    step._fill_dtype = np.dtype("float64")

    def build_feature_frame(self, branch_results: list[BranchResult]) -> pd.DataFrame:
        features: dict[str, float] = {}
        for branch in branch_results:
            for key, value in branch.details.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    features[key] = float(value)
        for column in self.feature_order:
            features.setdefault(column, np.nan)
        return pd.DataFrame([{column: features[column] for column in self.feature_order}])

    def predict(self, image: Image.Image) -> PredictionResult:
        image = normalize_pil_image(image)
        effnet = self.predict_effnet(image)
        vae = self.predict_vae(image)
        face_gate = self.detect_faces(image)
        openforensics = self.predict_openforensics(image, face_gate)
        semantic = self.predict_semantic(image)

        branch_list = [effnet, vae, face_gate, openforensics, semantic]
        features = self.build_feature_frame(branch_list)

        forensic_columns = list(self.schema.get("forensic_features", []))
        forensic_features = features.reindex(columns=forensic_columns)
        forensic_glm = self.load_forensic_glm()
        forensic_prob = float(forensic_glm.predict_proba(forensic_features)[0, 1])
        forensic_threshold = self.forensic_threshold()
        forensic_label = "Fake" if forensic_prob >= forensic_threshold else "Real"

        glm = self.load_glm()
        final_prob = float(glm.predict_proba(features)[0, 1])
        label = "Fake" if final_prob >= self.threshold else "Real"

        return PredictionResult(
            forensic_prob_fake=float(np.clip(forensic_prob, 0.0, 1.0)),
            forensic_label=forensic_label,
            forensic_threshold=forensic_threshold,
            final_prob_fake=float(np.clip(final_prob, 0.0, 1.0)),
            final_label=label,
            threshold=self.threshold,
            features=features,
            branches={
                "effnet": effnet,
                "vae": vae,
                "face_gate": face_gate,
                "openforensics": openforensics,
                "semantic": semantic,
            },
            image_size=image.size,
        )

    def load_metrics(self) -> dict:
        if self.paths.phase3_metrics.exists():
            return read_json(self.paths.phase3_metrics)
        return {}

    def load_coefficients(self) -> pd.DataFrame:
        if self.paths.phase3_coefficients.exists():
            return pd.read_csv(self.paths.phase3_coefficients)
        return pd.DataFrame()
