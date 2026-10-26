"""Save/recover the existing full-pipeline setup without creating another run.

Execute after section 3 once. In a new session, mount Drive and execute again
after installing the notebook dependencies. This restores setup, not optimizer
state. Every staged image is still checked against its recorded SHA-256.
"""
import hashlib
import importlib
import json
import re
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

SOURCE_FILES = (
    'src/__init__.py', 'src/ai_shield_adaptation.py',
    'src/ai_shield_adaptation_models.py', 'src/ai_shield_metrics.py',
    'src/ai_shield_cf_inference.py', 'src/ai_shield_full_adaptation.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_setup_state(namespace):
    required = ('CONFIG', 'RUN', 'frames', 'partition')
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError('Finish section 3 once before saving setup. Missing: ' + ', '.join(missing))
    run = Path(namespace['RUN']).resolve()
    root = Path(namespace['CONFIG'].drive_root).resolve()
    parent = root / 'AI_Shield_Adaptation/full_pipeline_v1'
    if run.parent != parent or not (run / 'experiment_plan.json').is_file():
        raise RuntimeError('RUN is not an existing full-pipeline experiment.')
    if json.loads((run / 'experiment_plan.json').read_text())['config'] != asdict(namespace['CONFIG']):
        raise RuntimeError('Current CONFIG differs from the saved experiment.')
    # These immutable small records identify the original experiment. No image
    # is reread here; preparation already checked it before creating this RUN.
    hashes = {name: digest(run / name) for name in (
        'training_roles.csv', 'input_hashes.json', 'experiment_plan.json')}
    sources = {name: digest(run / 'code' / name) for name in SOURCE_FILES}
    saved = {'version': 1, 'run_name': run.name, 'config': asdict(namespace['CONFIG']),
             'record_hashes': hashes, 'source_hashes': sources}
    atomic_json(run / 'setup_resume.json', saved)
    atomic_json(parent / 'active_setup.json', {'version': 1, 'run_name': run.name})
    print('SETUP SAVED:', run)
    if namespace.get('_AI_SHIELD_INTEGRATED'):
        print('Section 3 will reuse this saved experiment on the next setup recovery.')
    else:
        print('After a new runtime: mount Drive, restore dependencies if needed, and run this same cell instead of section 3.')
    return saved


def source_is_registered_patch(run, relative, before_hash, after_hash):
    if relative != 'src/ai_shield_adaptation_models.py':
        return False
    expected_path = (run / 'code' / relative).resolve()
    for record in run.glob('semantic_compatibility_fix_*.json'):
        try:
            payload = json.loads(record.read_text())
        except (ValueError, OSError):
            continue
        if payload.get('fix') != 'fast_processor_and_scoped_cudnn_tf32':
            continue
        for item in payload.get('files', []):
            if (Path(item['path']).resolve() == expected_path and
                    item['before_sha256'] == before_hash and item['after_sha256'] == after_hash):
                return True
    return False


def load_setup_record(root, run_name=None):
    parent = Path(root).resolve() / 'AI_Shield_Adaptation/full_pipeline_v1'
    if run_name is None:
        pointer = parent / 'active_setup.json'
        if not pointer.is_file():
            raise RuntimeError('No setup has been saved yet. Run section 3 once, then this cell.')
        run_name = json.loads(pointer.read_text())['run_name']
    if not re.fullmatch(r'\d{8}T\d{6}Z_[a-f0-9]{8}', run_name):
        raise ValueError('Invalid saved run name')
    run = (parent / run_name).resolve()
    if run.parent != parent:
        raise ValueError('Run path leaves the experiment folder')
    saved = json.loads((run / 'setup_resume.json').read_text())
    if saved.get('version') != 1 or saved['run_name'] != run_name:
        raise ValueError('Unexpected setup record')
    if Path(saved['config']['drive_root']).resolve() != Path(root).resolve():
        raise ValueError('This setup belongs to a different Drive root')
    for name in ('training_roles.csv', 'input_hashes.json', 'experiment_plan.json'):
        if digest(run / name) != saved['record_hashes'][name]:
            raise ValueError('Saved experiment record changed: ' + name)
    for relative in SOURCE_FILES:
        actual = digest(run / 'code' / relative)
        expected = saved['source_hashes'][relative]
        if actual != expected and not source_is_registered_patch(run, relative, expected, actual):
            raise ValueError('Saved source changed without a registered fix: ' + relative)
    return run, saved


def stage_verified_images(frames, destination):
    """Reuse verified local files; copy/hash missing files in the same read.

    A crash leaves at most an uncommitted partial file. Completed image files
    survive kernel-only restarts and can be reused without accessing Drive.
    """
    from tqdm.auto import tqdm
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stats = {'reused_local': 0, 'copied_from_drive': 0}
    staged = {}
    for split, frame in frames.items():
        paths = []
        for row in tqdm(frame.itertuples(), total=len(frame), desc='Restore ' + split):
            if not re.fullmatch(r'[a-f0-9]{24}', str(row.sample_id)):
                raise ValueError('Unexpected image ID')
            source = Path(row.local_path)
            target = destination / (str(row.sample_id) + source.suffix)
            if target.resolve().parent != destination:
                raise ValueError('Image path leaves the cache')
            if target.is_file() and digest(target) == row.sha256:
                stats['reused_local'] += 1
            else:
                temporary = target.with_name(target.name + '.partial')
                h = hashlib.sha256()
                # Only the helper's per-image temporary file is overwritten.
                with source.open('rb') as incoming, temporary.open('wb') as outgoing:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                        h.update(chunk)
                        outgoing.write(chunk)
                if h.hexdigest() != row.sha256:
                    raise ValueError('Image bytes differ from the recorded dataset: ' + str(row.sample_id))
                temporary.replace(target)
                stats['copied_from_drive'] += 1
            paths.append(str(target))
        staged[split] = frame.copy()
        staged[split]['local_path'] = paths
    return staged, stats


def restore_setup_state(namespace):
    root = Path(namespace.get('AI_SHIELD_DRIVE_ROOT', '/content/drive/MyDrive'))
    if not root.is_dir():
        raise RuntimeError("Mount Drive first: from google.colab import drive; drive.mount('/content/drive')")
    run, saved = load_setup_record(root, namespace.get('RESUME_RUN_NAME'))
    # Restoring arbitrary intermediate optimizer state is outside this helper.
    if any((run / name).exists() for name in (
        'effnet_adapted.pt', 'semantic_adapted.pt', 'selection_lock.json',
        'frozen_full_fusion.joblib', 'adapted_full_fusion.joblib')):
        raise RuntimeError('This run has progressed beyond setup. Its results are preserved; this helper cannot resume training/optimizer state.')
    if namespace.get('CONFIG') is not None and asdict(namespace['CONFIG']) != saved['config']:
        raise RuntimeError('Current CONFIG differs from the saved setup. Do not mix experiments.')
    try:
        import numpy as np
        import pandas as pd
        import torch
        import timm, transformers, sklearn
    except ImportError as error:
        raise RuntimeError('Restore section 1 dependencies in this runtime, then rerun this cell.') from error
    if not torch.cuda.is_available():
        raise RuntimeError('Reconnect to a GPU runtime before restoring this experiment.')
    # Load the exact saved implementation, including a registered semantic fix.
    code_root = run / 'code'
    for key in list(sys.modules):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]
    sys.path.insert(0, str(code_root))
    importlib.invalidate_caches()
    adaptation = importlib.import_module('src.ai_shield_adaptation')
    full = importlib.import_module('src.ai_shield_full_adaptation')
    models_module = importlib.import_module('src.ai_shield_adaptation_models')
    config = full.FullConfig(**saved['config'])
    frames = {}
    for split, expected in adaptation.SPLIT_HASHES.items():
        path = config.split_root / (split + '.csv')
        if digest(path) != expected:
            raise ValueError('Original dataset manifest changed: ' + split)
        frames[split] = adaptation.checked_csv(path)
    adaptation.validate_frames(frames, adaptation.EXPECTED_COUNTS)
    print('Restoring existing experiment:', run.name, flush=True)
    print('Checking original model files...', flush=True)
    full.verify_input_lock(config, run)
    partition = adaptation.checked_csv(run / 'training_roles.csv')
    expected_roles = full.full_partition(frames['train'], config)
    columns = ['sample_id', 'label', 'role']
    pd.testing.assert_frame_equal(partition[columns].reset_index(drop=True),
                                  expected_roles[columns].reset_index(drop=True))
    frames, stats = stage_verified_images(frames, Path('/content') / ('ai_shield_images_' + run.name))
    partition['local_path'] = partition.sample_id.map(frames['train'].set_index('sample_id').local_path)
    for module in (adaptation, full):
        namespace.update({k: v for k, v in vars(module).items() if not k.startswith('_')})
    from IPython.display import display
    audit = json.loads((config.audit_root / 'saved_pipeline_evaluation_report.json').read_text(encoding='utf-8'))
    namespace.update(CONFIG=config, RUN=run, frames=frames, partition=partition, audit=audit,
                     DEVICE='cuda', STAGE_TO_LOCAL=True, CODE_ROOT=code_root,
                     load_original_models=models_module.load_original_models,
                     np=np, pd=pd, torch=torch, display=display,
                     cache_auxiliary=reuse_complete_auxiliary)
    adaptation.seed_everything(config.seed)
    print('SETUP RESTORED:', stats)
    if namespace.get('_AI_SHIELD_INTEGRATED'):
        print('Continue with section 4. The compatibility check is included there.')
    else:
        print('Next: run the semantic repair if still needed, then restore_reference_checks(globals()).')
        print('Do not rerun section 3: your existing RUN and training roles are restored.')
    return stats


def restore_reference_checks(namespace):
    """Repeat small live GPU checks and keep all earlier reports intact."""
    from src.ai_shield_adaptation import check_audit_parity, write_json
    from src.ai_shield_adaptation_models import load_original_models
    from src.ai_shield_full_adaptation import FrozenAuxiliary, replay_original, verify_face_parity
    config, frames, run, device = (namespace[k] for k in ('CONFIG', 'frames', 'RUN', 'DEVICE'))
    reports = Path(run) / 'resume_checks' / uuid.uuid4().hex[:12]
    models, processor = load_original_models(config)
    parity = check_audit_parity(models, processor, frames, config, device)
    reports.mkdir(parents=True, exist_ok=False)
    write_json(reports / 'main_branch_parity.json', parity)
    replay = replay_original(config, frames, reports)
    auxiliary_model = FrozenAuxiliary(config)
    face_parity = verify_face_parity(auxiliary_model, config, frames, reports)
    namespace.update(models=models, processor=processor, parity=parity, replay=replay,
                     auxiliary_model=auxiliary_model, face_parity=face_parity)
    print('Reference checks passed. Continue with the auxiliary-cache cell in section 4.')


def reuse_complete_auxiliary(aux, frames, config, run):
    from src.ai_shield_adaptation import checked_csv, align_predictions
    from src.ai_shield_full_adaptation import AUX_FEATURES, cache_auxiliary
    lock = Path(run) / 'auxiliary_cache_hashes.json'
    if not lock.is_file():
        return cache_auxiliary(aux, frames, config, run)
    hashes = json.loads(lock.read_text())
    tables = {}
    for split, frame in frames.items():
        path = Path(run) / ('auxiliary_' + split + '.csv')
        if digest(path) != hashes[split]:
            raise ValueError('Auxiliary cache changed: ' + split)
        table = checked_csv(path)
        if set(table.split) != {split}:
            raise ValueError('Auxiliary cache has wrong split: ' + split)
        tables[split] = align_predictions(frame, table, AUX_FEATURES)
    print('Reused the complete, hash-verified auxiliary cache.')
    return tables


def main(namespace):
    # First use saves the current completed setup; subsequent empty sessions
    # recover it. No automatic "latest folder" guessing is used.
    if all(key in namespace for key in ('CONFIG', 'RUN', 'frames', 'partition')):
        save_setup_state(namespace)
    else:
        restore_setup_state(namespace)


def prepare_notebook_setup(namespace, embedded_sources, embedded_hashes):
    """Integrated section 3: keep this run, recover a saved setup, or start once."""
    from src.ai_shield_adaptation import checked_csv, SPLIT_HASHES, validate_frames, EXPECTED_COUNTS, write_json
    from src.ai_shield_full_adaptation import full_preflight, prepare_full_run
    config = namespace['CONFIG']
    # Historical comparisons need this report on every path, including an
    # already-loaded setup that returns without calling full_preflight again.
    namespace['audit'] = json.loads((config.audit_root / 'saved_pipeline_evaluation_report.json').read_text(encoding='utf-8'))
    parent = config.root / 'AI_Shield_Adaptation/full_pipeline_v1'
    current = namespace.get('RUN')
    requested = namespace.get('RESUME_RUN_NAME')
    if current is not None:
        selected = Path(current)
    elif requested:
        if not re.fullmatch(r'\d{8}T\d{6}Z_[a-f0-9]{8}', requested):
            raise ValueError('Invalid RESUME_RUN_NAME')
        selected = parent / requested
    elif (parent / 'active_setup.json').is_file():
        selected, _ = load_setup_record(config.root)
    else:
        # Adopt only an unambiguous existing experiment with the same settings.
        candidates = []
        for plan in parent.glob('*/experiment_plan.json'):
            if json.loads(plan.read_text()).get('config') == asdict(config):
                candidates.append(plan.parent)
        if len(candidates) > 1:
            names = ', '.join(sorted(path.name for path in candidates))
            raise RuntimeError('More than one saved experiment exists. Set RESUME_RUN_NAME in section 2 to the intended folder: ' + names)
        selected = candidates[0] if candidates else None

    if selected is not None:
        selected = selected.resolve()
        if selected.parent != parent.resolve() or not selected.is_dir():
            raise RuntimeError('Requested experiment folder is unavailable: ' + str(selected))
        if any((selected / name).exists() for name in (
            'effnet_adapted.pt', 'semantic_adapted.pt', 'selection_lock.json',
            'frozen_full_fusion.joblib', 'adapted_full_fusion.joblib')):
            raise RuntimeError('This experiment has progressed beyond setup. Its results are preserved; automatic optimizer/training resume is not implemented.')
        in_memory = (current is not None and all(key in namespace for key in ('frames', 'partition')))
        if not (selected / 'setup_resume.json').is_file():
            # Adopt an existing pre-training run without redoing a separate
            # full image scan. restore_setup_state checks bytes while staging.
            frames = {}
            for split, expected in SPLIT_HASHES.items():
                path = config.split_root / (split + '.csv')
                if digest(path) != expected:
                    raise ValueError('Original dataset manifest changed: ' + split)
                frames[split] = checked_csv(path)
            validate_frames(frames, EXPECTED_COUNTS)
            adopted = dict(namespace, RUN=selected, frames=frames,
                           partition=checked_csv(selected / 'training_roles.csv'))
            save_setup_state(adopted)
        namespace['RESUME_RUN_NAME'] = selected.name
        if in_memory:
            load_setup_record(config.root, selected.name)
            validate_frames(namespace['frames'], EXPECTED_COUNTS)
            recorded_roles = checked_csv(selected / 'training_roles.csv')
            import pandas as pd
            pd.testing.assert_frame_equal(
                namespace['partition'][['sample_id', 'label', 'role']].reset_index(drop=True),
                recorded_roles[['sample_id', 'label', 'role']].reset_index(drop=True))
            # The already completed section 3 has usable staged frames. An
            # interrupted copy leaves Drive paths instead, so restore that case.
            cache = Path('/content') / ('ai_shield_images_' + selected.name)
            all_staged = all(Path(p).parent == cache for frame in namespace['frames'].values() for p in frame.local_path)
            if all_staged:
                namespace['RUN'] = selected
                namespace['cache_auxiliary'] = reuse_complete_auxiliary
                save_setup_state(namespace)
                print('Reusing the setup already loaded in this session:', selected.name)
                return
        restore_setup_state(namespace)
        return

    frames, audit = full_preflight(config)
    run, partition = prepare_full_run(config, frames)
    write_json(run / 'embedded_source_hashes.json', embedded_hashes)
    for relative, source in embedded_sources.items():
        target = run / 'code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding='utf-8')
    namespace.update(RUN=run, frames=frames, audit=audit, partition=partition)
    # Persist before the copy: interrupted staging now has a known RUN to resume.
    save_setup_state(namespace)
    frames, stats = stage_verified_images(frames, Path('/content') / ('ai_shield_images_' + run.name))
    partition['local_path'] = partition.sample_id.map(frames['train'].set_index('sample_id').local_path)
    namespace.update(frames=frames, partition=partition, cache_auxiliary=reuse_complete_auxiliary)
    print('Setup ready:', stats)


if __name__ == '__main__':
    main(globals())
