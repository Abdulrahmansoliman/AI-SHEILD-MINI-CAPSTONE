"""Frozen Community Forensics inference, matching finetune_v1 evaluation."""
import hashlib
from pathlib import Path

import torch
from torch import nn
import timm
from PIL import Image, ImageOps
from torchvision import transforms


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CommunityForensicsDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False,
        )
        self.vit.head = nn.Linear(self.vit.num_features, 1)

    def forward(self, images):
        return self.vit(images)


class FrozenCF:
    def __init__(self, checkpoint_path, expected_sha256, threshold,
                 expected_manifest_sha256, device=None):
        if file_sha256(checkpoint_path) != expected_sha256:
            raise RuntimeError("Checkpoint SHA-256 mismatch.")
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1].")
        self.threshold = float(threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model = CommunityForensicsDetector()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("stage") != "stage2_last_blocks":
            raise RuntimeError("Checkpoint is not Stage 2.")
        if payload.get("experiment_manifest_sha256") != expected_manifest_sha256:
            raise RuntimeError("Checkpoint belongs to a different experiment manifest.")
        self.best_val_auc = float(payload["best_val_auc"])
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.requires_grad_(False)
        self.model.float().eval().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(440, interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def predict(self, path):
        with Image.open(path) as source:
            source.seek(0)
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor).reshape(-1)
            if logits.numel() != 1 or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid model output.")
            score = float(torch.sigmoid(logits)[0].cpu())
            logit = float(logits[0].cpu())
        return {"logit": logit, "prob_fake": score,
                "predicted_label": int(score >= self.threshold)}
