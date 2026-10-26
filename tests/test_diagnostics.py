"""Checks for data isolation, threshold boundaries and restart-safe diagnostics."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import ai_shield_diagnostics as d

ROOT = Path(__file__).resolve().parents[1]


def synthetic():
    rng = np.random.default_rng(191)
    def table(prefix, n, split):
        y = np.arange(n) % 2
        t = pd.DataFrame({'sample_id': [prefix + str(i) for i in range(n)], 'label': y, 'split': split})
        for key in d.FULL:
            x = rng.normal(size=n) + .9 * y
            t[key] = 1 / (1 + np.exp(-x)) if 'prob' in key else x
        return t
    train, val = table('tr', 80, 'train'), table('va', 40, 'validation')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=5000, random_state=191))
    model.fit(train[d.FULL], train.label)
    val['fusion_prob'] = model.predict_proba(val[d.FULL])[:, 1]
    return {'train': train, 'predictions': {'validation': val},
            'roles': train[['sample_id', 'label', 'split']].assign(role='fusion_train'),
            'selection': {'thresholds': {k: .5 for k in ('fusion_prob', 'effnet_prob', 'semantic_vit_prob')}},
            'cf': {'validation': val[['sample_id', 'label']].assign(prob_fake=val.fusion_prob)}, 'cf_threshold': .5}


class DiagnosticTests(unittest.TestCase):
    def test_float_threshold_round_trip(self):
        score = .36611184909624206
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'pred.csv'
            pd.DataFrame({'sample_id': ['a', 'b'], 'label': [1, 0], 'score': [score, .1]}).to_csv(p, index=False)
            frame = d.read_csv(p)
            self.assertEqual(frame.score.iloc[0], score)
            self.assertEqual(d.metrics(frame.label, frame.score, score)['tp'], 1)

    def test_align_uses_ids_and_rejects_partial_duplicate_or_wrong_labels(self):
        ref = pd.DataFrame({'sample_id': ['a', 'b'], 'label': [0, 1]})
        pred = ref.assign(score=[.1, .9])
        self.assertEqual(d.align(ref, pred.iloc[::-1], ['score']).score.tolist(), [.1, .9])
        for invalid in [pred.iloc[:1], pd.concat([pred.iloc[:1]] * 2), pred.assign(label=[1, 0]), pred.assign(score=[.1, np.nan])]:
            with self.assertRaises(ValueError):
                d.align(ref, invalid, ['score'])

    def test_low_fpr_respects_ties_and_can_predict_nothing(self):
        y = [0, 0, 1, 1]
        scores = np.array([.7, .7, .7, .9])
        threshold = d.low_fpr_threshold(y, scores)
        self.assertEqual(threshold, .9)
        self.assertEqual(d.metrics(y, scores, threshold)['fpr'], 0)
        all_equal = np.full(4, .6)
        self.assertGreater(d.low_fpr_threshold(y, all_equal), .6)

    def test_fit_does_not_require_test_data_and_reproduces_full_control(self):
        data = synthetic()
        models, lock, table = d.fit_ablations(data)
        self.assertEqual(len(models), 6)
        self.assertEqual(len(table), 6)
        self.assertLess(lock['full_refit_max_validation_error'], 1e-12)
        self.assertEqual(lock['selection_split'], 'validation')
        self.assertEqual(set(lock['fit_sample_ids']), set(data['train'].sample_id))

    def test_fit_rejects_leaked_training_ids(self):
        data = synthetic()
        data['train'].loc[0, 'sample_id'] = data['predictions']['validation'].sample_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'fitting IDs'):
            d.fit_ablations(data)

    def test_control_gate_detects_changed_saved_fusion(self):
        data = synthetic()
        data['predictions']['validation']['fusion_prob'] = .5
        with self.assertRaisesRegex(ValueError, 'Full-refit control'):
            d.fit_ablations(data)

    def test_evaluation_requires_saved_selection(self):
        data = synthetic()
        models, lock, _ = d.fit_ablations(data)
        with tempfile.TemporaryDirectory() as tmp:
            d.save_json(Path(tmp) / 'validation_selection.json', {'wrong': True})
            with self.assertRaisesRegex(ValueError, 'Save the validation selection'):
                d.evaluate_ablations(data, models, lock, tmp)

    def test_idempotent_evidence_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'evidence.json'
            d.save_json(p, {'x': 1})
            before = p.read_bytes()
            d.save_json(p, {'x': 1})
            self.assertEqual(before, p.read_bytes())
            with self.assertRaisesRegex(ValueError, 'differs'):
                d.save_json(p, {'x': 2})
            self.assertEqual(before, p.read_bytes())

    def test_group_leak_rejected(self):
        def frame(split, ids):
            return pd.DataFrame({'sample_id': ids, 'label': [0, 1], 'split': split,
                'sha256': [split+'0', split+'1'], 'group_id': [split+'0', split+'1'],
                'prompt_id': ['', split+'prompt'], 'local_path': [split+'0.jpg', split+'1.jpg']})
        manifests = {s: frame(s, [s+'0', s+'1']) for s in ('train', *d.SPLITS)}
        manifests['test'].loc[0, 'group_id'] = manifests['validation'].group_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'group_id leakage'):
            d.check_partitions(manifests, manifests['train'].assign(role='fusion_train'))

    def test_gallery_checks_hashes_and_handles_missing_images(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'
            Image.new('RGB', (10, 10), 'blue').save(path)
            rows = pd.DataFrame([{'sample_id': str(i), 'label': 0, 'case': 'both_wrong',
                'generator': 'test', 'CF_score': .8, 'saved_full_score': .7,
                'local_path': str(path) if i < 2 else str(Path(tmp)/'missing.png'),
                'sha256': d.sha(path) if i == 0 else '0'*64} for i in range(3)])
            image, statuses = d.render_gallery(rows, tmp)
            self.assertIsNotNone(image)
            self.assertEqual(statuses.status.tolist(), ['verified', 'hash_mismatch_not_displayed', 'missing_image'])

    def test_notebooks_are_valid_self_contained_and_cpu_only(self):
        import nbformat
        paths = sorted((ROOT / 'notebooks').glob('AI_Shield_0[123]_*_v1.ipynb'))
        self.assertEqual(len(paths), 3)
        expected = (ROOT / 'src/ai_shield_diagnostics.py').read_text(encoding='utf-8')
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                compile(cell.source, str(path), 'exec')
            setup = ast.parse(notebook.cells[2].source)
            assignments = {n.targets[0].id: n.value for n in setup.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
            self.assertEqual(ast.literal_eval(assignments['ANALYSIS_SOURCE']), expected)
            self.assertEqual(ast.literal_eval(assignments['EXPECTED_SOURCE_SHA256']), hashlib.sha256(expected.encode()).hexdigest())
            imports = [n for n in ast.walk(ast.parse(expected)) if isinstance(n, (ast.Import, ast.ImportFrom))]
            text = ' '.join(ast.unparse(n) for n in imports)
            self.assertNotIn('torch', text)
            self.assertNotIn('tensorflow', text)


if __name__ == '__main__':
    unittest.main()
