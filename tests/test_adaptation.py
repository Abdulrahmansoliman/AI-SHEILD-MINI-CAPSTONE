"""Safety and small CPU smoke tests; these are not real-data accuracy results."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ai_shield_adaptation import (
    Config, EXPECTED_COUNTS, FEATURES, SPLIT_HASHES, align_predictions,
    configure_training, fit_fusion, max_f1_threshold, partition_training,
    scored_metrics, train_branch, training_mode, validate_frames, write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def frames_fixture():
    frames = {}
    for split in EXPECTED_COUNTS:
        rows = []
        for i in range(40 if split == "train" else 8):
            identity = f"{split}-{i}"
            rows.append({"sample_id": identity, "label": i % 2, "split": split,
                "sha256": hashlib.sha256(identity.encode()).hexdigest(), "group_id": identity,
                "prompt_id": f"{split}-prompt-{i // 4}" if i % 2 else "",
                "generator": "gpt-image-2" if split == "unseen_generator" else "FLUX.2_max",
                "local_path": identity + ".jpg"})
        frames[split] = pd.DataFrame(rows)
    return frames


class DataSafetyTests(unittest.TestCase):
    def setUp(self):
        self.frames = frames_fixture()

    def test_valid_splits(self):
        validate_frames(self.frames)

    def test_reject_cross_split_identity_and_content(self):
        for column in ("sample_id", "group_id", "sha256", "local_path"):
            with self.subTest(column=column):
                frames = frames_fixture()
                frames["validation"].loc[0, column] = frames["train"].loc[0, column]
                with self.assertRaises(ValueError):
                    validate_frames(frames)

    def test_reject_empty_identity(self):
        self.frames["train"].loc[0, "sample_id"] = ""
        with self.assertRaises(ValueError):
            validate_frames(self.frames)

    def test_reject_prompt_leakage(self):
        self.frames["validation"].loc[1, "prompt_id"] = self.frames["train"].loc[1, "prompt_id"]
        with self.assertRaisesRegex(ValueError, "Prompt"):
            validate_frames(self.frames)

    def test_reject_heldout_generator_in_training(self):
        self.frames["train"].loc[1, "generator"] = "gpt-image-2"
        with self.assertRaisesRegex(ValueError, "Held-out"):
            validate_frames(self.frames)

    def test_reject_malformed_hash(self):
        self.frames["train"].loc[0, "sha256"] = "not-a-hash"
        with self.assertRaises(ValueError):
            validate_frames(self.frames)

    def test_reject_conflicting_labels_for_duplicate_bytes(self):
        self.frames["train"].loc[1, "sha256"] = self.frames["train"].loc[0, "sha256"]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_frames(self.frames)

    def test_partition_is_repeatable_and_preserves_groups(self):
        train = self.frames["train"].copy()
        train.loc[2, "sha256"] = train.loc[0, "sha256"]
        a = partition_training(train)
        b = partition_training(train)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.sample_id), set(train.sample_id))
        self.assertEqual(set(a.role), {"branch_train", "fusion_train"})
        for column in ("group_id", "sha256", "prompt_id"):
            selected = a[a[column].ne("")]
            self.assertTrue((selected.groupby(column).role.nunique() == 1).all())

    def test_partition_rejects_validation(self):
        with self.assertRaises(ValueError):
            partition_training(self.frames["validation"])

    def test_alignment_reorders_without_dropping_rows(self):
        frame = self.frames["test"]
        predictions = frame.assign(score=np.arange(len(frame)) / 10).iloc[::-1]
        aligned = align_predictions(frame, predictions, ["score"])
        self.assertEqual(aligned.sample_id.tolist(), frame.sample_id.tolist())
        self.assertEqual(aligned.score.tolist(), list(np.arange(len(frame)) / 10))

    def test_alignment_rejects_incomplete_mislabeled_and_nan(self):
        frame = self.frames["test"]
        predictions = frame.assign(score=0.5)
        bad_label = predictions.copy()
        bad_label.loc[0, "label"] = 1
        bad_score = predictions.copy()
        bad_score.loc[0, "score"] = np.nan
        for bad in (predictions.iloc[1:], bad_label, bad_score, pd.concat([predictions, predictions.iloc[:1]])):
            with self.assertRaises(ValueError):
                align_predictions(frame, bad, ["score"])

    def test_max_f1_uses_actual_scores_and_greater_equal(self):
        labels, scores = [0, 0, 1, 1], [0.1, 0.2, 0.633988, 0.9]
        threshold = max_f1_threshold(labels, scores)
        self.assertEqual(threshold, scores[2])
        self.assertEqual(scored_metrics(labels, scores, threshold)["f1"], 1.0)
        for invalid in ([0, 0, np.nan, 1], [0, 0, 2, 1]):
            with self.assertRaises(ValueError):
                max_f1_threshold(labels, invalid)

    def test_fusion_rejects_branch_training_overlap_and_test_data(self):
        predictions = self.frames["train"].copy()
        for i, feature in enumerate(FEATURES):
            predictions[feature] = predictions.label + i / 20
        with self.assertRaisesRegex(ValueError, "fine-tune"):
            fit_fusion(predictions, [predictions.sample_id.iloc[0]])
        with self.assertRaisesRegex(ValueError, "restricted"):
            fit_fusion(predictions.assign(split="test"), [])
        fusion = fit_fusion(predictions, [])
        self.assertTrue(np.isfinite(fusion.predict_proba(predictions[FEATURES])).all())
        np.testing.assert_allclose(fusion[0].mean_, predictions[FEATURES].mean().to_numpy())

    def test_write_once_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            write_json(path, {"original": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"changed": 2})
            self.assertEqual(json.loads(path.read_text()), {"original": 1})

    def test_recorded_split_hashes_are_valid(self):
        for value in SPLIT_HASHES.values():
            self.assertEqual(len(value), 64)


class NotebookTests(unittest.TestCase):
    def test_notebook_schema_syntax_and_embedded_sources(self):
        import nbformat
        notebook = nbformat.read(ROOT / "notebooks/AI_Shield_Forensic_Semantic_Adaptation_v1.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, cell.id, "exec")
                self.assertEqual(cell.outputs, [])
                if cell.source.startswith("EMBEDDED_SOURCES"):
                    tree = ast.parse(cell.source)
                    sources = ast.literal_eval(tree.body[0].value)
                    hashes = ast.literal_eval(tree.body[1].value)
                    for name, source in sources.items():
                        compile(source, name, "exec")
                        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), hashes[name])
                        if name != "src/__init__.py":
                            self.assertEqual(source, (ROOT / name).read_text())

    def test_semantic_head_matches_original_architecture(self):
        original = json.loads((ROOT / "notebooks/phase2_semantic_report.ipynb").read_text())
        source = next("".join(c["source"]) for c in original["cells"]
                      if c["cell_type"] == "code" and "".join(c["source"]).startswith("class CustomMultiHeadSelfAttention"))
        before = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.ClassDef)}
        after = {n.name: ast.dump(n) for n in ast.parse((ROOT / "src/ai_shield_adaptation_models.py").read_text()).body
                 if isinstance(n, ast.ClassDef)}
        for name in ("CustomMultiHeadSelfAttention", "SemanticTransformerBlock", "SemanticTransformerHead"):
            self.assertEqual(before[name], after[name])


class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_semantic_precision_is_scoped_and_restored_on_error(self):
        import torch
        from types import SimpleNamespace
        from src.ai_shield_adaptation_models import ViTSemanticAttentionClassifier

        class Backbone(torch.nn.Module):
            config = SimpleNamespace(hidden_size=8)
            fail = False

            def forward(inner, **kwargs):
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                if inner.fail:
                    raise RuntimeError("test failure")
                return SimpleNamespace(last_hidden_state=torch.zeros(1, 3, 8))

        backbone = Backbone()
        model = ViTSemanticAttentionClassifier(backbone).eval()
        fake_cuda_input = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(allow_tf32=False):
                model(fake_cuda_input)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                backbone.fail = True
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    model(fake_cuda_input)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul

    def test_real_effnet_frozen_batchnorm_and_trainable_classifier(self):
        import torch
        from src.ai_shield_adaptation_models import EfficientNetBinaryClassifier
        model = EfficientNetBinaryClassifier()
        head, backbone = configure_training(model, "effnet", "head")
        training_mode(model, "effnet", "head", head, backbone)
        before = model.model.bn1.running_mean.clone()
        model(torch.rand(2, 3, 256, 256)).sum().backward()
        self.assertTrue(torch.equal(before, model.model.bn1.running_mean))
        self.assertIsNone(model.model.conv_stem.weight.grad)
        self.assertIsNotNone(model.model.classifier.weight.grad)
        head, backbone = configure_training(model, "effnet", "partial")
        training_mode(model, "effnet", "partial", head, backbone)
        self.assertFalse(model.model.bn1.training)
        self.assertTrue(model.model.bn2.training)

    def test_semantic_head_and_limited_backbone_gradients(self):
        import torch
        from transformers import ViTConfig, ViTModel
        from src.ai_shield_adaptation_models import ViTSemanticAttentionClassifier
        backbone = ViTModel(ViTConfig(image_size=32, patch_size=16, hidden_size=32,
                                     num_hidden_layers=2, num_attention_heads=4, intermediate_size=64))
        model = ViTSemanticAttentionClassifier(backbone)
        head, backbone_modules = configure_training(model, "semantic", "head")
        training_mode(model, "semantic", "head", head, backbone_modules)
        out = model(torch.rand(2, 3, 32, 32))
        out["logits"].sum().backward()
        self.assertTrue(torch.isfinite(out["logits"]).all())
        self.assertTrue(all(p.grad is None for p in model.vit.parameters()))
        self.assertIsNotNone(head.classifier[-1].weight.grad)
        configure_training(model, "semantic", "partial")
        self.assertTrue(all(p.requires_grad for p in model.vit.encoder.layer[-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.vit.encoder.layer[0].parameters()))

    def test_training_loop_writes_selected_checkpoint_on_synthetic_images(self):
        import torch
        from torch import nn
        from PIL import Image
        from src.ai_shield_adaptation_models import EfficientNetBinaryClassifier
        with tempfile.TemporaryDirectory() as tmp:
            frames = frames_fixture()
            for split in ("train", "validation"):
                frame = frames[split].iloc[:4].copy()
                for index in frame.index:
                    path = Path(tmp) / (frame.loc[index, "sample_id"] + ".png")
                    Image.new("RGB", (32, 32), (20 + 100 * int(frame.loc[index, "label"]), 50, 70)).save(path)
                    frame.loc[index, "local_path"] = str(path)
                frames[split] = frame
            model = EfficientNetBinaryClassifier()
            config = Config(head_epochs=1, partial_epochs=0, batch_size=2)
            report = train_branch(model, "effnet", None, frames["train"], frames["validation"], config, tmp, "cpu")
            self.assertEqual(report["training_images"], 4)
            self.assertTrue((Path(tmp) / "effnet_adapted.pt").exists())
            self.assertEqual(len(pd.read_csv(Path(tmp) / "effnet_history.csv")), 1)
            with self.assertRaises(FileExistsError):
                train_branch(model, "effnet", None, frames["train"], frames["validation"], config, tmp, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
