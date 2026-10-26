"""Recovery must preserve evidence and verify actual bytes, even after a crash."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('setup_resume', ROOT / 'scripts/AI_Shield_Setup_Resume_v1.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class StagingTests(unittest.TestCase):
    def fixture(self, root):
        path = root / 'original.jpg'
        path.write_bytes(b'original image bytes')
        frame = pd.DataFrame([{'sample_id': 'a' * 24, 'local_path': str(path),
                               'sha256': resume.digest(path), 'label': 0, 'split': 'test'}])
        return path, {'test': frame}

    def test_verified_local_copy_reused_without_drive_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, frames = self.fixture(root)
            staged, stats = resume.stage_verified_images(frames, root / 'cache')
            self.assertEqual(stats, {'reused_local': 0, 'copied_from_drive': 1})
            original.unlink()
            again, stats = resume.stage_verified_images(frames, root / 'cache')
            self.assertEqual(stats, {'reused_local': 1, 'copied_from_drive': 0})
            pd.testing.assert_frame_equal(staged['test'], again['test'])

    def test_corrupt_cached_file_repaired_from_verified_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, frames = self.fixture(root)
            staged, _ = resume.stage_verified_images(frames, root / 'cache')
            target = Path(staged['test'].local_path.iloc[0])
            target.write_bytes(b'corrupt')
            _, stats = resume.stage_verified_images(frames, root / 'cache')
            self.assertEqual(stats['copied_from_drive'], 1)
            self.assertEqual(resume.digest(target), frames['test'].sha256.iloc[0])

    def test_bad_source_never_committed_as_valid_local_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, frames = self.fixture(root)
            original.write_bytes(b'changed original')
            with self.assertRaisesRegex(ValueError, 'Image bytes differ'):
                resume.stage_verified_images(frames, root / 'cache')
            self.assertFalse((root / 'cache' / ('a' * 24 + '.jpg')).exists())

    def test_partial_copy_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, frames = self.fixture(root)
            cache = root / 'cache'
            cache.mkdir()
            (cache / ('a' * 24 + '.jpg.partial')).write_bytes(b'half-written')
            _, stats = resume.stage_verified_images(frames, cache)
            self.assertEqual(stats['copied_from_drive'], 1)
            self.assertFalse((cache / ('a' * 24 + '.jpg.partial')).exists())

    def test_invalid_id_cannot_leave_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, frames = self.fixture(root)
            frames['test']['sample_id'] = '../outside'
            with self.assertRaises(ValueError):
                resume.stage_verified_images(frames, root / 'cache')


class SnapshotTests(unittest.TestCase):
    def fixture(self, root):
        from src.ai_shield_full_adaptation import FullConfig
        from dataclasses import asdict
        config = FullConfig(drive_root=str(root))
        config.audit_root.mkdir(parents=True)
        (config.audit_root / 'saved_pipeline_evaluation_report.json').write_text(json.dumps({'metrics': [], 'marker': 'original-audit'}))
        run = root / 'AI_Shield_Adaptation/full_pipeline_v1/20261026T132144Z_58c9dad3'
        run.mkdir(parents=True)
        (run / 'experiment_plan.json').write_text(json.dumps({'config': asdict(config)}))
        for name in ('training_roles.csv', 'input_hashes.json'):
            (run / name).write_text('original')
        for name in resume.SOURCE_FILES:
            path = run / 'code' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original ' + name)
        return {'CONFIG': config, 'RUN': run, 'frames': {}, 'partition': None}

    def test_save_and_restore_identifies_same_run_without_creating_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace = self.fixture(root)
            resume.save_setup_state(namespace)
            actual, saved = resume.load_setup_record(root)
            self.assertEqual(actual, namespace['RUN'])
            self.assertEqual(saved['config']['seed'], 191)
            self.assertEqual(len(list(actual.parent.glob('2026*'))), 1)

    def test_changed_training_roles_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace = self.fixture(root)
            resume.save_setup_state(namespace)
            (namespace['RUN'] / 'training_roles.csv').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'record changed'):
                resume.load_setup_record(root)

    def test_only_registered_semantic_fix_can_change_saved_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace = self.fixture(root)
            resume.save_setup_state(namespace)
            run = namespace['RUN']
            path = run / 'code/src/ai_shield_adaptation_models.py'
            before = resume.digest(path)
            path.write_text('patched semantic model')
            with self.assertRaisesRegex(ValueError, 'Saved source changed'):
                resume.load_setup_record(root)
            (run / 'semantic_compatibility_fix_test.json').write_text(json.dumps({
                'fix': 'fast_processor_and_scoped_cudnn_tf32',
                'files': [{'path': str(path), 'before_sha256': before, 'after_sha256': resume.digest(path)}]}))
            actual, _ = resume.load_setup_record(root)
            self.assertEqual(actual, run)

    def test_completed_auxiliary_cache_is_reused_and_corruption_stops(self):
        from src.ai_shield_full_adaptation import AUX_FEATURES
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            frame = pd.DataFrame([{'sample_id': 'sample', 'label': 0, 'split': 'test'}])
            table = frame.assign(**{key: .25 for key in AUX_FEATURES})
            path = run / 'auxiliary_test.csv'
            table.to_csv(path, index=False)
            (run / 'auxiliary_cache_hashes.json').write_text(json.dumps({'test': resume.digest(path)}))
            with patch('src.ai_shield_full_adaptation.cache_auxiliary', side_effect=AssertionError('Should not recompute')):
                actual = resume.reuse_complete_auxiliary(None, {'test': frame}, None, run)
                self.assertEqual(len(actual['test']), 1)
                path.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'cache changed'):
                    resume.reuse_complete_auxiliary(None, {'test': frame}, None, run)

    def test_integrated_setup_reuses_saved_pointer_without_new_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved_namespace = self.fixture(root)
            resume.save_setup_state(saved_namespace)
            namespace = {'CONFIG': saved_namespace['CONFIG']}
            with patch.object(resume, 'restore_setup_state') as restore, \
                    patch('src.ai_shield_full_adaptation.full_preflight', side_effect=AssertionError('Repeated scan')), \
                    patch('src.ai_shield_full_adaptation.prepare_full_run', side_effect=AssertionError('New run')):
                resume.prepare_notebook_setup(namespace, {}, {})
            restore.assert_called_once_with(namespace)
            self.assertEqual(namespace['RESUME_RUN_NAME'], saved_namespace['RUN'].name)
            self.assertEqual(namespace['audit']['marker'], 'original-audit')

    def test_integrated_setup_does_not_choose_between_multiple_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved_namespace = self.fixture(root)
            first = saved_namespace['RUN']
            other = first.parent / '20261026T142144Z_12345678'
            other.mkdir()
            (other / 'experiment_plan.json').write_bytes((first / 'experiment_plan.json').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'More than one saved experiment'):
                resume.prepare_notebook_setup({'CONFIG': saved_namespace['CONFIG']}, {}, {})

    def test_integrated_setup_refuses_to_overwrite_training_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            namespace = self.fixture(Path(tmp))
            checkpoint = namespace['RUN'] / 'semantic_adapted.pt'
            checkpoint.write_bytes(b'keep trained checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'beyond setup'):
                resume.prepare_notebook_setup(namespace, {}, {})
            self.assertEqual(checkpoint.read_bytes(), b'keep trained checkpoint')

    def test_integrated_setup_keeps_completed_in_memory_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace = self.fixture(root)
            frame = pd.DataFrame([{'sample_id': 'a' * 24, 'label': 0, 'role': 'branch_train',
                'local_path': str(Path('/content') / ('ai_shield_images_' + namespace['RUN'].name) / ('a' * 24 + '.jpg'))}])
            frame.to_csv(namespace['RUN'] / 'training_roles.csv', index=False)
            namespace.update(frames={'train': frame}, partition=frame)
            resume.save_setup_state(namespace)
            with patch('src.ai_shield_adaptation.validate_frames'), \
                    patch.object(resume, 'restore_setup_state', side_effect=AssertionError('Repeated restore')), \
                    patch('src.ai_shield_full_adaptation.full_preflight', side_effect=AssertionError('Repeated scan')):
                resume.prepare_notebook_setup(namespace, {}, {})
            self.assertIs(namespace['frames']['train'], frame)
            self.assertEqual(namespace['audit']['marker'], 'original-audit')

    def test_integrated_setup_adopts_one_existing_run_without_new_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.fixture(root)
            config = original['CONFIG']
            config.split_root.mkdir(parents=True)
            frame = pd.DataFrame([{'sample_id': 'a' * 24, 'label': 0, 'role': 'branch_train'}])
            frame.to_csv(config.split_root / 'train.csv', index=False)
            frame.to_csv(original['RUN'] / 'training_roles.csv', index=False)
            namespace = {'CONFIG': config}
            with patch('src.ai_shield_adaptation.SPLIT_HASHES', {'train': resume.digest(config.split_root / 'train.csv')}), \
                    patch('src.ai_shield_adaptation.validate_frames'), \
                    patch.object(resume, 'restore_setup_state') as restore, \
                    patch('src.ai_shield_full_adaptation.prepare_full_run', side_effect=AssertionError('New run')):
                resume.prepare_notebook_setup(namespace, {}, {})
            restore.assert_called_once_with(namespace)
            self.assertTrue((original['RUN'] / 'setup_resume.json').is_file())
            self.assertEqual(namespace['audit']['marker'], 'original-audit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
