"""Full-pipeline data isolation, reproducibility and standalone inference checks."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.ai_shield_adaptation import FEATURES, predict_pair, sha256_file, write_json
from src.ai_shield_full_adaptation import (
    AUX_FEATURES, FULL_FEATURES, FullConfig, FullPredictor, FrozenAuxiliary,
    combine_features, fit_full_models, full_partition, latent_seed,
    load_legacy_fusion, lock_full_selection, replay_original, stage_images, verify_full_lock,
)
from tests.test_adaptation import frames_fixture

ROOT = Path(__file__).resolve().parents[1]


def score_fixture(prefix, n=32):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'sample_id': [f'{prefix}-{i}' for i in range(n)],
                          'label': np.arange(n) % 2, 'split': 'train'})
    for name in FEATURES:
        frame[name] = rng.normal(size=n) + .4 * frame.label
    aux = frame[['sample_id', 'label']].copy()
    for name in AUX_FEATURES:
        aux[name] = rng.random(n)
    return frame, aux


class FullPipelineTests(unittest.TestCase):
    def test_exact_seventeen_feature_schema(self):
        self.assertEqual(len(FULL_FEATURES), 17)
        self.assertEqual(len(set(FULL_FEATURES)), 17)
        self.assertEqual(FULL_FEATURES[:4], ['effnet_logit', 'effnet_prob',
                         'effnet_calibrated_prob', 'vae_recon_error'])
        self.assertEqual(FULL_FEATURES[11:], FEATURES[2:])

    def test_three_way_partition_preserves_connected_groups(self):
        train = pd.concat([frames_fixture()['train'].assign(
            sample_id=lambda f: f.sample_id + f'-{k}',
            group_id=lambda f: f.group_id + f'-{k}',
            prompt_id=lambda f: f.prompt_id.where(f.prompt_id.eq(''), f.prompt_id + f'-{k}'),
            sha256=lambda f: f.sha256.map(lambda x: hashlib.sha256((x + str(k)).encode()).hexdigest()))
            for k in range(5)], ignore_index=True)
        train.loc[2, 'sha256'] = train.loc[0, 'sha256']
        split = full_partition(train, FullConfig())
        pd.testing.assert_frame_equal(split, full_partition(train, FullConfig()))
        self.assertEqual(set(split.role), {'branch_train', 'calibration', 'fusion_train'})
        self.assertEqual(set(split.sample_id), set(train.sample_id))
        for column in ('group_id', 'sha256', 'prompt_id'):
            self.assertTrue(split[split[column].ne('')].groupby(column).role.nunique().eq(1).all())
        self.assertTrue(split.groupby('role').label.nunique().eq(2).all())

    def test_full_fit_uses_separate_calibration_and_seventeen_inputs(self):
        calibration, _ = score_fixture('cal')
        fusion, aux = score_fixture('fusion')
        cal, model, combined = fit_full_models(calibration, fusion, aux.iloc[::-1], ['branch'])
        self.assertEqual(list(model.feature_names_in_), FULL_FEATURES)
        self.assertEqual(list(cal.feature_names_in_), ['effnet_logit'])
        self.assertTrue(np.isfinite(model.predict_proba(combined[FULL_FEATURES])).all())
        np.testing.assert_array_equal(combined[AUX_FEATURES], aux[AUX_FEATURES])
        changed = fusion.assign(effnet_logit=fusion.effnet_logit + 10)
        refreshed = combine_features(changed, aux, cal)
        self.assertFalse(np.allclose(combined.effnet_calibrated_prob, refreshed.effnet_calibrated_prob))
        np.testing.assert_array_equal(combined[AUX_FEATURES], refreshed[AUX_FEATURES])

    def test_fitting_rejects_test_images_role_overlap_and_nan(self):
        cal, _ = score_fixture('cal')
        fusion, aux = score_fixture('fusion')
        with self.assertRaisesRegex(ValueError, 'training data only'):
            fit_full_models(cal.assign(split='test'), fusion, aux, ['branch'])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            fit_full_models(cal, fusion, aux, [cal.sample_id.iloc[0]])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            fit_full_models(fusion, fusion, aux, ['branch'])
        with self.assertRaises(ValueError):
            fit_full_models(cal.assign(effnet_logit=np.nan), fusion, aux, ['branch'])

    def test_combine_rejects_missing_duplicate_or_wrong_label_auxiliary(self):
        cal, _ = score_fixture('cal')
        fusion, aux = score_fixture('fusion')
        calibrator, _, _ = fit_full_models(cal, fusion, aux, ['branch'])
        bad_label = aux.copy()
        bad_label.loc[0, 'label'] = 1
        for bad in [aux.iloc[1:], pd.concat([aux, aux.iloc[:1]]), bad_label]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                combine_features(fusion, bad, calibrator)

    def test_replay_checks_owned_fusion_scores(self):
        cal, _ = score_fixture('cal')
        fusion, aux = score_fixture('fusion')
        _, model, pred = fit_full_models(cal, fusion, aux, ['branch'])
        pred['aishield_full_prob'] = model.predict_proba(pred[FULL_FEATURES])[:, 1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            joblib.dump(model, root / 'legacy.pkl')
            frames = {}
            for split in ('test', 'unseen_generator'):
                copy = pred.assign(split=split)
                copy.to_csv(root / f'aishield_{split}_predictions.csv', index=False)
                frames[split] = copy
            config = SimpleNamespace(audit_root=root, extras={'full_glm': root / 'legacy.pkl'})
            report = replay_original(config, frames, root)
            self.assertEqual([r['n'] for r in report], [32, 32])
            self.assertLess(max(r['max_absolute_error'] for r in report), 1e-12)
            pred.assign(aishield_full_prob=0).to_csv(root / 'aishield_test_predictions.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                replay_original(config, frames, root)

    def test_validation_lock_detects_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['effnet_adapted.pt', 'semantic_adapted.pt', 'input_hashes.json',
                     'training_roles.csv', 'auxiliary_cache_hashes.json']
            names += [f'auxiliary_{s}.csv' for s in ('train', 'validation', 'test', 'unseen_generator')]
            selections = {'frozen_full': {'validation_metrics': {'f1': .7, 'auc': .8}},
                          'adapted_full': {'validation_metrics': {'f1': .75, 'auc': .8}}}
            for variant in selections:
                names += [f'{variant}_{s}' for s in ('calibrator.joblib', 'fusion.joblib', 'selection.json')]
            for name in names:
                (root / name).write_text(name)
            lock = lock_full_selection(root, selections)
            self.assertEqual(lock['validation_selected_candidate'], 'adapted_full')
            self.assertEqual(verify_full_lock(root), lock)
            (root / 'effnet_adapted.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_full_lock(root)

    def test_staging_preserves_bytes_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'original.jpg'
            source.write_bytes(b'fixture bytes')
            frame = pd.DataFrame([{'sample_id': 'image-1', 'local_path': str(source), 'sha256': sha256_file(source)}])
            staged = stage_images({'test': frame}, root / 'staged')
            self.assertEqual(Path(staged['test'].local_path.iloc[0]).read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                stage_images({'test': frame}, root / 'staged')

    def test_sampled_vae_is_repeatable_without_changing_global_rng(self):
        import torch
        class DummyVAE:
            def encode(self, x):
                return torch.zeros(1, 2), torch.zeros(1, 2)
            def decode(self, z):
                return z.sum().sigmoid().expand(1, 3, 2, 2)
        aux = FrozenAuxiliary.__new__(FrozenAuxiliary)
        aux.vae = DummyVAE()
        aux.vae_transform = lambda image: torch.zeros(3, 2, 2)
        aux.reference = np.linspace(0, 1, 101)
        aux.face_features = lambda image: {}
        state = torch.random.get_rng_state().clone()
        a = aux.features(None, 'same-bytes')
        b = aux.features(None, 'same-bytes')
        self.assertEqual(a, b)
        self.assertNotEqual(a, aux.features(None, 'other-bytes'))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertNotEqual(latent_seed('a'), latent_seed('b'))

    def test_full_standalone_matches_batched_features(self):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        torch.set_num_threads(2)
        class Eff(torch.nn.Module):
            def forward(self, x):
                self.last_batch_size = len(x)
                return x.mean((1, 2, 3))
        class Semantic(torch.nn.Module):
            def forward(self, x):
                self.last_batch_size = len(x)
                score = x.mean((1, 2, 3))
                return {'logits': score, **{key: score * .1 for key in
                    ('attention_entropy', 'attention_focus', 'self_attention_entropy', 'self_attention_focus')}}
        def processor(images, return_tensors):
            return {'pixel_values': to_tensor(images.resize((32, 32)))[None]}
        class Aux:
            def features(self, image, identity):
                return {name: .25 for name in AUX_FEATURES}
        cal, _ = score_fixture('cal')
        fusion, aux = score_fixture('fusion')
        calibrator, combiner, _ = fit_full_models(cal, fusion, aux, ['branch'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'
            Image.new('RGB', (40, 30), (100, 20, 70)).save(path)
            frame = pd.DataFrame([{'sample_id': 'image', 'label': 1, 'split': 'test', 'local_path': str(path)}])
            models = {'effnet': Eff(), 'semantic': Semantic()}
            main = predict_pair(models, processor, frame, FullConfig(), 'cpu')
            aux_frame = frame[['sample_id', 'label']].assign(**{name: .25 for name in AUX_FEATURES})
            full = combine_features(main, aux_frame, calibrator)
            expected = combiner.predict_proba(full[FULL_FEATURES])[0, 1]
            predictor = FullPredictor(models, processor, Aux(), calibrator, combiner, 'cpu')
            self.assertAlmostEqual(expected, predictor.predict(path), places=12)
            self.assertEqual(models['effnet'].last_batch_size, 1)
            self.assertEqual(models['semantic'].last_batch_size, 16)
            self.assertEqual(predictor.semantic_batch_size, 16)
            custom = FullPredictor(models, processor, Aux(), calibrator, combiner, 'cpu', semantic_batch_size=4)
            self.assertAlmostEqual(expected, custom.predict(path), places=12)
            self.assertEqual(models['semantic'].last_batch_size, 4)
            for invalid in (0, -1, 1.5, True):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    FullPredictor(models, processor, Aux(), calibrator, combiner, 'cpu', semantic_batch_size=invalid)

    def test_notebook_schema_code_and_embedded_sources(self):
        import nbformat
        notebook = nbformat.read(ROOT / 'notebooks/AI_Shield_Full_Pipeline_FineTuning_v1.ipynb', as_version=4)
        nbformat.validate(notebook)
        found = False
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            compile(cell.source, cell.id, 'exec')
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            if cell.source.startswith('EMBEDDED_SOURCES ='):
                assignments = ast.parse(cell.source).body[:2]
                sources = ast.literal_eval(assignments[0].value)
                digests = ast.literal_eval(assignments[1].value)
                for name, source in sources.items():
                    self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digests[name])
                    if name != 'src/__init__.py':
                        self.assertEqual(source, (ROOT / name).read_text(encoding='utf-8'))
                self.assertIn('src/ai_shield_full_adaptation.py', sources)
                found = True
        self.assertTrue(found)

    def test_setup_supports_python313_and_checks_before_uninstall(self):
        notebook = json.loads((ROOT / 'notebooks/AI_Shield_Full_Pipeline_FineTuning_v1.ipynb').read_text())
        code = ''.join(notebook['cells'][2]['source'])
        for fail_resolution in (False, True):
            with self.subTest(fail_resolution=fail_resolution), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def fake_run(command, **kwargs):
                    calls.append(command)
                    failed = fail_resolution and '--dry-run' in command
                    return SimpleNamespace(returncode=int(failed), stdout='resolver failure' if failed else 'ok')
                real_path = type(Path(tmp))
                with patch('sys.version_info', (3, 13, 1)), \
                     patch('pathlib.Path', side_effect=lambda p: real_path(tmp) / real_path(p).name), \
                     patch('importlib.metadata.version', return_value='2.0.0'), \
                     patch('subprocess.run', side_effect=fake_run):
                    if fail_resolution:
                        with self.assertRaisesRegex(RuntimeError, 'Checking dependency compatibility failed'):
                            exec(compile(code, '<setup>', 'exec'), {})
                    else:
                        exec(compile(code, '<setup>', 'exec'), {})
                self.assertIn('--dry-run', calls[0])
                self.assertIn('tensorflow-cpu==2.20.0', calls[0])
                self.assertNotIn('tensorflow-cpu==2.19.0', calls[0])
                if fail_resolution:
                    self.assertEqual(len(calls), 1)
                    self.assertNotIn('uninstall', calls[0])
                else:
                    self.assertIn('uninstall', calls[1])
                    self.assertIn('install', calls[2])
                    self.assertIn('IMPORT CHECK PASSED', calls[3][-1])

    def test_integrated_setup_calls_embedded_recovery_and_records_live_parity(self):
        notebook = json.loads((ROOT / 'notebooks/AI_Shield_Full_Pipeline_FineTuning_v1.ipynb').read_text())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        calls = []
        namespace = {
            'setup_tools': {'prepare_notebook_setup': lambda ns, sources, hashes: calls.append(('setup', ns))},
            'EMBEDDED_SOURCES': {}, 'EMBEDDED_HASHES': {},
        }
        setup_first_line = cells['full-adapt-07'].splitlines()[0]
        exec(setup_first_line, namespace)
        self.assertEqual(calls[0], ('setup', namespace))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            def verified_fix(ns):
                calls.append('live-parity')
                ns.update(models={}, processor=None, parity={'n': 6})
            namespace.update({
                'RUN': run, 'CONFIG': None, 'frames': {}, 'DEVICE': 'cuda',
                'compatibility_tools': {'apply_fix': verified_fix},
                'setup_tools': {'save_setup_state': lambda ns: calls.append('save')},
                'write_json': lambda path, payload: calls.append(('write', path)),
                'replay_original': lambda *a: [], 'FrozenAuxiliary': lambda c: object(),
                'verify_face_parity': lambda *a: [1] * 6,
                'display': lambda x: None, 'pd': pd,
            })
            exec(cells['full-adapt-09'], namespace)
            self.assertLess(calls.index('live-parity'), calls.index('save'))
            self.assertIn(('write', run / 'main_branch_parity.json'), calls)

    def test_comparison_cell_loads_audit_without_setup_variable(self):
        notebook = json.loads((ROOT / 'notebooks/AI_Shield_Full_Pipeline_FineTuning_v1.ipynb').read_text())
        code = ''.join(next(c for c in notebook['cells'] if c['id'] == 'full-adapt-17')['source'])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            expected_audit = {'metrics': [], 'marker': 'saved-audit'}
            (run / 'saved_pipeline_evaluation_report.json').write_text(json.dumps(expected_audit))
            metric = {'model': 'candidate', 'split': 'test', 'auc': .8, 'f1': .7,
                      'accuracy': .75, 'fpr': .1, 'threshold': .5}
            def references(frames, config, audit, folder):
                self.assertEqual(audit, expected_audit)
                return [dict(metric, model='reference')]
            namespace = dict(json=json, pd=pd, RUN=run, CONFIG=SimpleNamespace(audit_root=run),
                selections={}, models={}, processor=None, frames={}, auxiliary={},
                adapted_calibrator=None, adapted_fusion=None, adapted_thresholds={},
                frozen_calibrator=None, frozen_fusion=None, frozen_thresholds={}, DEVICE='cpu',
                lock_full_selection=lambda *args: {'validation_selected_candidate': 'adapted_full'},
                verify_full_lock=lambda *args: None, verify_input_lock=lambda *args: None,
                evaluate_full_variant=lambda *args: ([dict(metric, model=args[0])], []),
                load_original_models=lambda *args: ({}, None), historical_references=references,
                gc=SimpleNamespace(collect=lambda: None),
                torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
                display=lambda *args: None)
            self.assertNotIn('audit', namespace)
            exec(code, namespace)
            self.assertEqual(len(namespace['comparison']), 3)
            self.assertTrue((run / 'comparison.csv').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
