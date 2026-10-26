"""Verify fixed-size semantic inference against this trained run before saving.

No training, threshold changes, replacement predictions, or relaxed tolerances.
The extra semantic work is part of the measured single-request latency.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path

OLD_INIT = '    def __init__(self,models,processor,auxiliary,calibrator,fusion,device):\n        from src.ai_shield_adaptation import PairPredictor'
NEW_INIT = '''    def __init__(self,models,processor,auxiliary,calibrator,fusion,device,semantic_batch_size=16):
        from src.ai_shield_adaptation import PairPredictor
        if isinstance(semantic_batch_size, bool) or not isinstance(semantic_batch_size, int) or semantic_batch_size < 1:
            raise ValueError('semantic_batch_size must be a positive integer')
        self.semantic_batch_size = semantic_batch_size'''
OLD_FORWARD = "            sem = pair.models['semantic'](pair.processor(images=image,return_tensors='pt')['pixel_values'].to(pair.device))"
NEW_FORWARD = '''            # Match evaluation's physical batch size; keep only the first result.
            # cuDNN/ViT numerical paths can differ for a physical batch of one.
            pixels = pair.processor(images=image,return_tensors='pt')['pixel_values']
            if pixels.shape[0] != 1:
                raise ValueError('Standalone inference expects one image')
            pixels = pixels.repeat(self.semantic_batch_size, 1, 1, 1).to(pair.device)
            sem = pair.models['semantic'](pixels)'''
OLD_LOAD = "    return FullPredictor(models,processor,FrozenAuxiliary(config),joblib.load(Path(run)/f'{variant}_calibrator.joblib'),joblib.load(Path(run)/f'{variant}_fusion.joblib'),device)"
NEW_LOAD = '''    saved_batch_size = int(json.loads((Path(run)/'experiment_plan.json').read_text())['config']['batch_size'])
    return FullPredictor(models,processor,FrozenAuxiliary(config),joblib.load(Path(run)/f'{variant}_calibrator.joblib'),joblib.load(Path(run)/f'{variant}_fusion.joblib'),device,semantic_batch_size=saved_batch_size)'''


def patched_source(source):
    for old, new in ((OLD_INIT, NEW_INIT), (OLD_FORWARD, NEW_FORWARD), (OLD_LOAD, NEW_LOAD)):
        if new in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError('Unrecognized standalone source. No source files changed.')
        source = source.replace(old, new, 1)
    compile(source, 'standalone_batch_candidate', 'exec')
    return source


def check_candidate(namespace, candidate_class, batch_size):
    import numpy as np
    import pandas as pd
    from src.ai_shield_adaptation import checked_csv, sha256_file
    from src.ai_shield_full_adaptation import FULL_FEATURES
    fusion = namespace['adapted_fusion']

    class Capture:
        def predict_proba(self, values):
            self.values = values.iloc[0].copy()
            return fusion.predict_proba(values)

    capture = Capture()
    candidate = candidate_class(namespace['models'], namespace['processor'], namespace['auxiliary_model'],
        namespace['adapted_calibrator'], capture, namespace['DEVICE'], semantic_batch_size=batch_size)
    test = namespace['frames']['test'].sort_values('sample_id').groupby('label', group_keys=False).head(3)
    probes = [('test', test)]
    unseen = namespace['frames'].get('unseen_generator')
    if unseen is not None and len(unseen) % batch_size:
        probes.append(('unseen_generator', unseen.tail(len(unseen) % batch_size)))
    records = []
    for split, frame in probes:
        saved = checked_csv(Path(namespace['RUN']) / f'adapted_full_{split}_predictions.csv').set_index('sample_id')
        for row in frame.itertuples():
            if sha256_file(row.local_path) != row.sha256:
                raise ValueError('Reference image bytes changed: ' + row.sample_id)
            expected = saved.loc[row.sample_id]
            score = candidate.predict(row.local_path)
            live = capture.values[FULL_FEATURES].to_numpy(dtype=float)
            old = expected[FULL_FEATURES].to_numpy(dtype=float)
            feature_match = bool(np.allclose(live, old, atol=5e-4, rtol=1e-4))
            score_match = bool(np.isclose(score, expected.fusion_prob, atol=5e-4, rtol=1e-4))
            records.append({'sample_id': row.sample_id, 'split': split, 'saved_score': float(expected.fusion_prob),
                'standalone_score': score, 'absolute_error': abs(score-float(expected.fusion_prob)),
                'max_feature_error': float(np.max(np.abs(live-old))), 'features_match': feature_match, 'score_matches': score_match})
    table = pd.DataFrame(records)
    if 'display' in namespace:
        namespace['display'](table)
    if not records or not all(r['features_match'] and r['score_matches'] for r in records):
        raise ValueError('Fixed-batch inference did not reproduce every reference. No source files were changed; share the table above.')
    return records


def apply_standalone_batch_fix(namespace):
    import src.ai_shield_full_adaptation as current
    required = ('CONFIG', 'RUN', 'frames', 'models', 'processor', 'auxiliary_model',
                'adapted_calibrator', 'adapted_fusion', 'DEVICE')
    missing = [key for key in required if key not in namespace]
    if missing:
        raise RuntimeError('Keep the trained session. Missing variables: ' + ', '.join(missing))
    run = Path(namespace['RUN'])
    batch_size = int(json.loads((run / 'experiment_plan.json').read_text())['config']['batch_size'])
    if batch_size != namespace['CONFIG'].batch_size:
        raise ValueError('Current CONFIG differs from the saved evaluation batch size')
    current.verify_full_lock(run)
    paths = list(dict.fromkeys([Path(current.__file__).resolve(), (run / 'code/src/ai_shield_full_adaptation.py').resolve()]))
    plans = [(path, path.read_bytes(), patched_source(path.read_text(encoding='utf-8'))) for path in paths]
    tree = ast.parse(plans[0][2])
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FullPredictor')
    candidate_namespace = dict(vars(current))
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), 'standalone_batch_candidate', 'exec'), candidate_namespace)
    print('Checking standalone inference with semantic internal batch size', batch_size, '...', flush=True)
    checks = check_candidate(namespace, candidate_namespace['FullPredictor'], batch_size)

    # Save only after all unchanged feature and score tolerances pass.
    changes = []
    for path, before, source in plans:
        after = source.encode('utf-8')
        before_hash, after_hash = hashlib.sha256(before).hexdigest(), hashlib.sha256(after).hexdigest()
        backup = path.with_name(path.name + '.before-standalone-fix-' + before_hash[:12])
        if before != after:
            if backup.exists() and backup.read_bytes() != before:
                raise RuntimeError('Backup conflict: ' + str(backup))
            if not backup.exists():
                backup.write_bytes(before)
            temporary = path.with_name(path.name + '.standalone-fix-tmp')
            temporary.write_bytes(after)
            temporary.replace(path)
        changes.append({'path': str(path), 'before_sha256': before_hash, 'after_sha256': after_hash})
    record = {'fix': 'fixed_semantic_batch_for_standalone', 'logical_request_batch_size': 1,
              'semantic_internal_batch_size': batch_size, 'atol': 5e-4, 'rtol': 1e-4,
              'checks': checks, 'files': changes,
              'cost_note': 'Repeats the input inside the semantic branch. Include this extra compute in latency and memory measurements.'}
    encoded = json.dumps(record, indent=2).encode('utf-8')
    report_path = run / ('standalone_batch_compatibility_' + hashlib.sha256(encoded).hexdigest()[:12] + '.json')
    if not report_path.exists():
        report_path.write_bytes(encoded)
    importlib.invalidate_caches()
    importlib.reload(current)
    namespace['FullPredictor'] = current.FullPredictor
    namespace['load_full_predictor'] = current.load_full_predictor
    namespace['semantic_internal_batch_size'] = batch_size
    print('PASS:', len(checks), 'reference images; feature and score checks passed at the original tolerance.')
    print('Weights, thresholds, saved predictions and the selection lock were not changed.')
    print('Continuing section 8. Timing includes the repeated semantic computation.')
    return checks


if __name__ == '__main__':
    apply_standalone_batch_fix(globals())
