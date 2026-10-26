"""Controlled two-branch adaptation experiment. Never imported by the app."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ai_shield_metrics import binary_metrics


EXPECTED_COUNTS = {"train": 2990, "validation": 640, "test": 640, "unseen_generator": 854}
SPLIT_HASHES = {
    "train": "29e893703856c3e1c5e71a0a5f04619ef01bb0e7cc8829a01e39ce605fc9f2a2",
    "validation": "1d69e4c55ab1896b51421b2e9bbc3d1a919966087ca7601f8f3f3919250aaf11",
    "test": "fa53c8831a4d4657d1ea34558b284d13dc65935686255761f89638c6802af063",
    "unseen_generator": "3fc0d4ea61bab0340b9512e732042bddb3f066768aadb9c1dbc574c05fb8d392",
}
SEMANTIC_ID = "google/vit-base-patch16-224-in21k"
SEMANTIC_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
FEATURES = [
    "effnet_logit", "effnet_prob", "semantic_vit_logit", "semantic_vit_prob",
    "semantic_attention_entropy", "semantic_attention_focus",
    "semantic_self_attention_entropy", "semantic_self_attention_focus",
]


@dataclass(frozen=True)
class Config:
    drive_root: str = "/content/drive/MyDrive"
    seed: int = 191
    fusion_fraction: float = 0.20
    head_epochs: int = 3
    partial_epochs: int = 3
    head_lr: float = 3e-4
    partial_head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-2
    batch_size: int = 16
    num_workers: int = 0

    @property
    def root(self):
        return Path(self.drive_root)

    @property
    def split_root(self):
        return self.root / "AI_Shield_Data/splits/v1"

    @property
    def cf_root(self):
        return self.root / "AI_Shield_CommunityForensics/experiments/finetune_v1"

    @property
    def audit_root(self):
        return self.root / "AI_Shield_Baseline_Evaluation/saved_pipeline_v1/results"

    @property
    def checkpoints(self):
        return {
            "effnet": self.root / "AI_Shield_Phase1/effnet_b0_tinygenimage_full_balanced_v1/effnet_b0_tinygenimage_full_balanced_v1_best.pt",
            "semantic": self.root / "AI_Shield_Phase2_Semantic/semantic_vit_transformer_attention_v1/checkpoints/semantic_vit_transformer_attention_best.pt",
        }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    # Exclusive creation makes a repeated cell fail instead of replacing evidence.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checked_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={
        "sample_id": str, "sha256": str, "group_id": str, "prompt_id": str,
    })


def validate_frames(frames, expected_counts=None):
    if set(frames) != set(EXPECTED_COUNTS):
        raise ValueError("Expected train, validation, test and unseen_generator splits.")
    required = {"sample_id", "label", "split", "sha256", "group_id", "prompt_id", "generator", "local_path"}
    for name, frame in frames.items():
        if not required.issubset(frame):
            raise ValueError(f"{name}: missing columns {required - set(frame)}")
        if expected_counts and len(frame) != expected_counts[name]:
            raise ValueError(f"{name}: wrong sample count")
        if set(frame.label) != {0, 1} or set(frame.split) != {name}:
            raise ValueError(f"{name}: invalid labels or split values")
        for column in ("sample_id", "group_id", "sha256", "local_path"):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{name}: empty {column}")
        if not frame.sha256.str.fullmatch(r"[a-f0-9]{64}").all():
            raise ValueError(f"{name}: malformed SHA-256")
    all_rows = pd.concat(frames.values(), ignore_index=True)
    if all_rows.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    for column in ("sha256", "group_id", "local_path"):
        if (all_rows.groupby(column).split.nunique() > 1).any():
            raise ValueError(f"{column} leakage across outer splits")
    if (all_rows.groupby("sha256").label.nunique() > 1).any():
        raise ValueError("Identical image bytes have conflicting labels")
    dev = all_rows[all_rows.split != "unseen_generator"]
    fake = dev[dev.label == 1]
    if fake.generator.eq("gpt-image-2").any():
        raise ValueError("Held-out generator entered development")
    prompts = fake[fake.prompt_id.astype(str).str.strip().ne("")]
    if (prompts.groupby("prompt_id").split.nunique() > 1).any():
        raise ValueError("Prompt leakage across development splits")
    unseen = frames["unseen_generator"]
    if set(unseen.loc[unseen.label == 1, "generator"]) != {"gpt-image-2"}:
        raise ValueError("Unexpected held-out generator")


def partition_training(train, fraction=0.20, seed=191):
    if set(train.split) != {"train"}:
        raise ValueError("Inner partition must only use the original training split")
    if not 0 < fraction < 1:
        raise ValueError("Fusion fraction must be in (0, 1)")
    train = train.reset_index(drop=True)
    # Connected groups keep prompts, source groups and duplicate bytes together.
    parent = list(range(len(train)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for column in ("group_id", "sha256", "prompt_id"):
        seen = {}
        for index, row in train.iterrows():
            value = str(row[column]).strip()
            if not value or (column == "prompt_id" and row.label != 1):
                continue
            if value in seen:
                parent[find(index)] = find(seen[value])
            else:
                seen[value] = index
    groups = [find(i) for i in range(len(train))]
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    branch, fusion = next(splitter.split(train, train.label, groups))
    result = train.copy()
    result["role"] = "branch_train"
    result.loc[fusion, "role"] = "fusion_train"
    for role, subset in result.groupby("role"):
        if set(subset.label) != {0, 1}:
            raise ValueError(f"{role} needs both classes; do not silently resplit")
    return result


def align_predictions(reference, predictions, columns):
    required = {"sample_id", "label", *columns}
    if not required.issubset(predictions):
        raise ValueError(f"Missing prediction columns: {required - set(predictions)}")
    if reference.sample_id.duplicated().any() or predictions.sample_id.duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    if set(reference.sample_id) != set(predictions.sample_id):
        raise ValueError("Prediction IDs differ from the manifest; no partial comparisons")
    ordered = predictions.set_index("sample_id").loc[reference.sample_id].reset_index()
    if not np.array_equal(reference.label.to_numpy(), ordered.label.to_numpy()):
        raise ValueError("Prediction labels do not match manifest")
    if not np.isfinite(ordered[columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite predictions")
    return ordered


def max_f1_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Need both classes and finite [0, 1] scores")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def fit_fusion(predictions, branch_training_ids):
    if set(predictions.split) != {"train"}:
        raise ValueError("Fusion fitting is restricted to training images")
    if set(predictions.sample_id) & set(branch_training_ids):
        raise ValueError("Fusion images were used to fine-tune a branch")
    align_predictions(predictions, predictions, FEATURES)
    if set(predictions.label) != {0, 1}:
        raise ValueError("Fusion needs both classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=191))
    model.fit(predictions[FEATURES], predictions.label)
    return model


def preflight(config):
    frames = {}
    for name in EXPECTED_COUNTS:
        path = config.split_root / f"{name}.csv"
        if sha256_file(path) != SPLIT_HASHES[name]:
            raise ValueError(f"{name}: differs from the recorded CF split")
        frames[name] = checked_csv(path)
    validate_frames(frames, EXPECTED_COUNTS)
    audit = json.loads((config.audit_root / "saved_pipeline_evaluation_report.json").read_text())
    for branch, path in config.checkpoints.items():
        if sha256_file(path) != audit["source_artifacts"][f"{branch}_sha256"]:
            raise ValueError(f"{branch}: checkpoint differs from the saved baseline audit")
    from tqdm.auto import tqdm
    for row in tqdm(pd.concat(frames.values()).itertuples(), total=sum(EXPECTED_COUNTS.values()), desc="Verify image bytes"):
        if sha256_file(row.local_path) != row.sha256:
            raise ValueError(f"Changed image: {row.sample_id}")
    return frames, audit


def prepare_run(config, frames):
    if config.head_epochs < 1 or config.partial_epochs < 0 or config.batch_size < 1:
        raise ValueError("Invalid training budget")
    partition = partition_training(frames["train"], config.fusion_fraction, config.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run = config.root / "AI_Shield_Adaptation" / "two_branch_v1" / run_id
    run.mkdir(parents=True, exist_ok=False)
    partition.to_csv(run / "training_roles.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "timm", "transformers", "scikit-learn", "pandas", "numpy", "Pillow",
    )}
    write_json(run / "experiment_plan.json", {
        "config": asdict(config), "libraries": versions, "split_sha256": SPLIT_HASHES,
        "source_sha256": {k: sha256_file(v) for k, v in config.checkpoints.items()},
        "implementation_sha256": sha256_file(__file__),
        "semantic_revision": SEMANTIC_REVISION, "feature_order": FEATURES,
        "training_roles_sha256": sha256_file(run / "training_roles.csv"),
        "counts_by_role": partition.role.value_counts().to_dict(),
        "scope": "Two-branch variant, NOT the original 17-feature full pipeline. No VAE/face branch.",
        "budget_note": "Same 2990-image development budget as CF; branch training uses only the branch_train subset. CF trained on all 2990.",
        "preprocessing": "Original audit: frame 0, RGB without EXIF transpose; EffNet resize256 bilinear/ImageNet; semantic pinned HF processor. Train-only horizontal flip 0.5.",
        "test_policy": "Previously examined historical tests; never used for fitting/selection. Fresh confirmation still needed.",
        "pretraining_cost": "Not measured; training times cover only this adaptation, not original pretraining.",
    })
    return run, partition


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def open_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        image.seek(0)
        return image.convert("RGB").copy()


def make_loader(frame, branch, processor, config, training=False):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class Images(Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, index):
            from PIL import Image
            row = frame.iloc[index]
            image = open_rgb(row.local_path)
            if training and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = transform(image) if branch == "effnet" else processor(images=image, return_tensors="pt")["pixel_values"][0]
            return x, float(row.label), str(row.sample_id)

    # Zero workers is deliberate: the notebook-local dataset also works on Windows.
    if config.num_workers != 0:
        raise ValueError("This portable loader uses num_workers=0")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(Images(), batch_size=config.batch_size, shuffle=training, num_workers=0, generator=generator)


def predict_branch(model, loader, frame, branch, device):
    import torch
    model.to(device).eval()
    rows = []
    with torch.inference_mode():
        for x, labels, ids in loader:
            result = model(x.to(device))
            logits = result if branch == "effnet" else result["logits"]
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite model logits")
            scores = torch.sigmoid(logits).cpu().tolist()
            for i, sample_id in enumerate(ids):
                prefix = "effnet" if branch == "effnet" else "semantic_vit"
                row = {"sample_id": sample_id, "label": int(labels[i]),
                       f"{prefix}_logit": float(logits[i].cpu()), f"{prefix}_prob": scores[i]}
                if branch == "semantic":
                    for key in ("attention_entropy", "attention_focus", "self_attention_entropy", "self_attention_focus"):
                        row[f"semantic_{key}"] = float(result[key][i].cpu())
                rows.append(row)
    columns = FEATURES[:2] if branch == "effnet" else FEATURES[2:]
    result = align_predictions(frame, pd.DataFrame(rows), columns)
    result["split"] = frame.split.to_numpy()
    return result


def predict_pair(models, processor, frame, config, device):
    results = []
    for branch, model in models.items():
        loader = make_loader(frame, branch, processor, config)
        results.append(predict_branch(model, loader, frame, branch, device))
        model.cpu()
    merged = results[0].merge(results[1], on=["sample_id", "label", "split"], validate="one_to_one")
    return align_predictions(frame, merged, FEATURES)


def check_audit_parity(models, processor, frames, config, device):
    # This is a fixed implementation check, not a metric used for model selection.
    fixture = frames["test"].sort_values("sample_id").groupby("label", group_keys=False).head(3)
    actual = predict_pair(models, processor, fixture, config, device)
    saved = checked_csv(config.audit_root / "aishield_test_predictions.csv")
    saved = saved[saved.sample_id.isin(fixture.sample_id)]
    saved = align_predictions(fixture, saved, FEATURES)
    differences = {}
    for column in FEATURES:
        a, b = actual[column].to_numpy(), saved[column].to_numpy()
        differences[column] = float(np.max(np.abs(a - b)))
        if not np.allclose(a, b, atol=5e-4, rtol=1e-4):
            raise ValueError(f"Saved baseline parity failed: {column}. Stop before training; investigate preprocessing/backbone versions.")
    return {"n": len(fixture), "sample_ids": fixture.sample_id.tolist(), "max_absolute_errors": differences}


def configure_training(model, branch, stage):
    model.requires_grad_(False)
    if branch == "effnet":
        head = model.model.classifier
        backbone = [model.model.blocks[-1], model.model.conv_head, model.model.bn2]
    else:
        head = model.semantic_head
        backbone = [model.vit.encoder.layer[-1], model.vit.layernorm]
    head.requires_grad_(True)
    if stage == "partial":
        for module in backbone:
            module.requires_grad_(True)
    elif stage != "head":
        raise ValueError("Unknown training stage")
    return head, backbone


def training_mode(model, branch, stage, head, backbone):
    # requires_grad=False alone would not freeze BatchNorm running statistics.
    model.eval()
    head.train()
    if branch == "effnet":
        model.model.training = True  # classifier's functional dropout only
    if stage == "partial":
        for module in backbone:
            module.train()


def train_branch(model, branch, processor, branch_frame, validation, config, run, device):
    import torch
    from sklearn.metrics import roc_auc_score
    if set(branch_frame.split) != {"train"} or set(validation.split) != {"validation"}:
        raise ValueError("Wrong training/selection split")
    seed_everything(config.seed)
    checkpoint = Path(run) / f"{branch}_adapted.pt"
    if checkpoint.exists():
        raise FileExistsError("Start a new run instead of overwriting a trained branch")
    model.to(device)
    train_loader = make_loader(branch_frame, branch, processor, config, training=True)
    val_loader = make_loader(validation, branch, processor, config)
    score_column = "effnet_prob" if branch == "effnet" else "semantic_vit_prob"
    initial = predict_branch(model, val_loader, validation, branch, device)
    best_auc = float(roc_auc_score(initial.label, initial[score_column]))
    selected = {"stage": "original", "epoch": 0, "validation_auc": best_auc}
    torch.save({"model_state": model.state_dict(), **selected}, checkpoint)
    history, max_trainable = [], 0
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for stage, epochs in (("head", config.head_epochs), ("partial", config.partial_epochs)):
        if epochs == 0:
            continue
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state"], strict=True)
        head, backbone = configure_training(model, branch, stage)
        head_ids = {id(p) for p in head.parameters()}
        groups = [{"params": list(head.parameters()), "lr": config.head_lr if stage == "head" else config.partial_head_lr}]
        others = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        if others:
            groups.append({"params": others, "lr": config.backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        max_trainable = max(max_trainable, trainable)
        for epoch in range(1, epochs + 1):
            training_mode(model, branch, stage, head, backbone)
            loss_sum, count = 0.0, 0
            epoch_start = time.perf_counter()
            for x, labels, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(x.to(device))
                logits = output if branch == "effnet" else output["logits"]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device, dtype=torch.float32))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                count += len(labels)
            predictions = predict_branch(model, val_loader, validation, branch, device)
            auc = float(roc_auc_score(predictions.label, predictions[score_column]))
            row = {"branch": branch, "stage": stage, "epoch": epoch, "train_loss": loss_sum / count,
                   "validation_auc": auc, "seconds_including_validation": time.perf_counter() - epoch_start,
                   "trainable_parameters": trainable}
            history.append(row)
            print(row)
            pd.DataFrame(history).to_csv(Path(run) / f"{branch}_history.csv", index=False)
            if auc > best_auc:
                best_auc = auc
                selected = {"stage": stage, "epoch": epoch, "validation_auc": auc}
                torch.save({"model_state": model.state_dict(), **selected}, checkpoint)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state"], strict=True)
    model.cpu().eval()
    report = {"branch": branch, "selected": selected, "training_images": len(branch_frame),
              "adaptation_seconds_including_validation_and_saves": elapsed,
              "max_trainable_parameters": max_trainable, "total_parameters": sum(p.numel() for p in model.parameters()),
              "cuda_peak_allocated_bytes": peak, "checkpoint_sha256": sha256_file(checkpoint)}
    write_json(Path(run) / f"{branch}_training.json", report)
    return report


def fit_variant(name, models, processor, partition, validation, config, run, device):
    import joblib
    fusion_rows = partition[partition.role == "fusion_train"]
    branch_ids = partition.loc[partition.role == "branch_train", "sample_id"]
    train_predictions = predict_pair(models, processor, fusion_rows, config, device)
    validation_predictions = predict_pair(models, processor, validation, config, device)
    start = time.perf_counter()
    fusion = fit_fusion(train_predictions, branch_ids)
    fit_seconds = time.perf_counter() - start
    validation_predictions["fusion_prob"] = fusion.predict_proba(validation_predictions[FEATURES])[:, 1]
    thresholds = {key: max_f1_threshold(validation_predictions.label, validation_predictions[key])
                  for key in ("effnet_prob", "semantic_vit_prob", "fusion_prob")}
    write_json(Path(run) / f"{name}_selection.json", {
        "thresholds": thresholds, "selection_split": "validation", "fusion_fit_seconds": fit_seconds,
        "fusion_training_images": len(fusion_rows), "feature_order": FEATURES,
    })
    joblib.dump(fusion, Path(run) / f"{name}_fusion.joblib")
    train_predictions.to_csv(Path(run) / f"{name}_fusion_training_predictions.csv", index=False)
    validation_predictions.to_csv(Path(run) / f"{name}_validation_predictions.csv", index=False)
    return fusion, thresholds


def scored_metrics(labels, scores, threshold):
    if not np.isfinite(scores).all() or ((np.asarray(scores) < 0) | (np.asarray(scores) > 1)).any():
        raise ValueError("Invalid detection scores")
    result = binary_metrics(labels, scores, threshold)
    result["fpr"] = result["fp"] / (result["fp"] + result["tn"])
    return result


def evaluate_variant(name, models, processor, frames, fusion, thresholds, config, run, device):
    rows = []
    for split in ("test", "unseen_generator"):
        predictions = predict_pair(models, processor, frames[split], config, device)
        predictions["fusion_prob"] = fusion.predict_proba(predictions[FEATURES])[:, 1]
        predictions.to_csv(Path(run) / f"{name}_{split}_predictions.csv", index=False)
        for column in ("effnet_prob", "semantic_vit_prob", "fusion_prob"):
            rows.append({"model": f"{name}/{column}", "split": split,
                         **scored_metrics(predictions.label, predictions[column], thresholds[column])})
    return rows


def historical_references(frames, config, audit, run):
    rows, evidence = [], {}
    threshold_path = config.cf_root / "results/validation_thresholds.json"
    thresholds = json.loads(threshold_path.read_text())
    evidence[str(threshold_path)] = sha256_file(threshold_path)
    old_metrics = [r for r in audit["metrics"] if "Full fusion" in r["model"]]
    for split in ("test", "unseen_generator"):
        for key in ("pretrained", "finetuned"):
            path = config.cf_root / f"predictions/{key}_{split}.csv"
            pred = align_predictions(frames[split], checked_csv(path), ["prob_fake"])
            threshold = thresholds["baseline" if key == "pretrained" else key]["max_f1"]
            rows.append({"model": f"CF/{key}", "split": split,
                         **scored_metrics(pred.label, pred.prob_fake, threshold)})
            evidence[str(path)] = sha256_file(path)
        matches = [r for r in old_metrics if r["split"] == split]
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous original full fusion reference")
        path = config.audit_root / f"aishield_{split}_predictions.csv"
        pred = align_predictions(frames[split], checked_csv(path), ["aishield_full_prob"])
        rows.append({"model": "original_full_17_feature/reference_only", "split": split,
                     **scored_metrics(pred.label, pred.aishield_full_prob, matches[0]["threshold"])})
        evidence[str(path)] = sha256_file(path)
    write_json(Path(run) / "historical_reference_hashes.json", evidence)
    return rows


def benchmark(predict_one, paths, device, repeats=3, warmup=3):
    """Warm batch-1 end-to-end latency; excludes checkpoint/model loading."""
    import torch
    import psutil
    if not paths or repeats < 1:
        raise ValueError("Need benchmark images and positive repeats")
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    for path in paths[:warmup]:
        predict_one(path)
    sync()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(repeats):
        for path in paths:
            sync()
            start = time.perf_counter()
            predict_one(path)
            sync()
            durations.append(time.perf_counter() - start)
    return {"images": len(paths), "repeats": repeats, "batch_size": 1,
            "median_seconds": float(np.median(durations)), "p95_seconds": float(np.quantile(durations, .95)),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
            "process_rss_after_bytes": psutil.Process().memory_info().rss,
            "memory_note": "RSS is the whole notebook process after this run, NOT isolated model peak memory.",
            "scope": "warm file-open + preprocessing + transfer + model + score; no model load, API, provenance or network"}


class PairPredictor:
    def __init__(self, models, processor, fusion, device):
        from torchvision import transforms
        self.models, self.processor, self.fusion, self.device = models, processor, fusion, device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for model in models.values():
            model.to(device).eval()

    def predict(self, path):
        import torch
        image = open_rgb(path)
        with torch.inference_mode():
            logit = self.models["effnet"](self.transform(image).unsqueeze(0).to(self.device))[0]
            semantic = self.models["semantic"](self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device))
            row = {"effnet_logit": float(logit), "effnet_prob": float(torch.sigmoid(logit)),
                   "semantic_vit_logit": float(semantic["logits"][0]),
                   "semantic_vit_prob": float(torch.sigmoid(semantic["logits"][0]))}
            for key in ("attention_entropy", "attention_focus", "self_attention_entropy", "self_attention_focus"):
                row["semantic_" + key] = float(semantic[key][0])
        values = pd.DataFrame([row])[FEATURES]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Non-finite inference features")
        return float(self.fusion.predict_proba(values)[0, 1])
