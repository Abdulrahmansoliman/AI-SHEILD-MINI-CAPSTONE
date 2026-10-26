"""Fixed-shape compatibility must be verified, never achieved by loosening a gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ai_shield_adaptation import sha256_file
from src.ai_shield_full_adaptation import FULL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('standalone_batch_fix', ROOT / 'scripts/AI_Shield_Standalone_Batch_Fix_v1.py')
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


class StandaloneFixTests(unittest.TestCase):
    def test_patch_matches_exported_inference_and_is_idempotent(self):
        current = (ROOT / 'src/ai_shield_full_adaptation.py').read_text()
        old = current.replace(fix.NEW_INIT, fix.OLD_INIT).replace(fix.NEW_FORWARD, fix.OLD_FORWARD).replace(fix.NEW_LOAD, fix.OLD_LOAD)
        self.assertEqual(fix.patched_source(old), current)
        self.assertEqual(fix.patched_source(current), current)
        with self.assertRaisesRegex(RuntimeError, 'Unrecognized'):
            fix.patched_source('print("unknown source")')

    def fixture(self, root):
        image = root / 'image.jpg'
        image.write_bytes(b'owned fixture bytes')
        rows = []
        for split, count in (('test', 8), ('unseen_generator', 6)):
            frame = pd.DataFrame([{'sample_id': f'{split}-{i}', 'label': i % 2,
                'local_path': str(image), 'sha256': sha256_file(image), 'split': split} for i in range(count)])
            saved = frame[['sample_id', 'label']].assign(**{key: .25 for key in FULL_FEATURES}, fusion_prob=.4)
            saved.to_csv(root / f'adapted_full_{split}_predictions.csv', index=False)
            rows.append((split, frame))
        class Fusion:
            def predict_proba(self, values):
                return np.array([[.6, .4]])
        return {'RUN': root, 'frames': dict(rows), 'models': {}, 'processor': None,
            'auxiliary_model': None, 'adapted_calibrator': None, 'adapted_fusion': Fusion(), 'DEVICE': 'cpu'}

    def test_gate_covers_six_reference_images_and_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            namespace = self.fixture(Path(tmp))
            class Candidate:
                def __init__(self, models, processor, auxiliary, calibrator, fusion, device, semantic_batch_size):
                    self.fusion = fusion
                    self.size = semantic_batch_size
                def predict(self, path):
                    assert self.size == 16
                    return float(self.fusion.predict_proba(pd.DataFrame([{key: .25 for key in FULL_FEATURES}]))[0, 1])
            records = fix.check_candidate(namespace, Candidate, 16)
            self.assertEqual(len(records), 12)
            self.assertTrue(all(row['features_match'] and row['score_matches'] for row in records))

    def test_gate_rejects_feature_drift_even_when_final_score_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            namespace = self.fixture(Path(tmp))
            class Candidate:
                def __init__(self, models, processor, auxiliary, calibrator, fusion, device, semantic_batch_size):
                    self.fusion = fusion
                def predict(self, path):
                    values = {key: .25 for key in FULL_FEATURES}
                    values['semantic_vit_logit'] += .01
                    return float(self.fusion.predict_proba(pd.DataFrame([values]))[0, 1])
            with self.assertRaisesRegex(ValueError, 'did not reproduce every reference'):
                fix.check_candidate(namespace, Candidate, 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
