"""CPU-only analysis of a completed run. Never imports or changes image models."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix, brier_score_loss)

VERSION = '1.0.0'
DEFAULT_RUN = '20261026T132144Z_58c9dad3'
SEM = ['semantic_vit_logit', 'semantic_vit_prob', 'semantic_attention_entropy',
       'semantic_attention_focus', 'semantic_self_attention_entropy',
       'semantic_self_attention_focus']
MAIN = ['effnet_logit', 'effnet_prob'] + SEM
VAE = ['vae_recon_error', 'vae_recon_percentile']
FACE = ['face_present', 'num_faces', 'largest_face_ratio', 'face_confidence',
        'openforensics_applicable', 'openforensics_prob_fake']
FULL = MAIN[:2] + ['effnet_calibrated_prob'] + VAE + FACE + SEM
SPLITS = ('validation', 'test', 'unseen_generator')
# Fixed before inspecting new test results. No hyperparameter sweep.
ABLATIONS = {
    'semantic_features': SEM,
    'two_logits': ['effnet_logit', 'semantic_vit_logit'],
    'two_branches': MAIN,
    'full_without_vae': [x for x in FULL if x not in VAE],
    'full_without_face': [x for x in FULL if x not in FACE],
    'full_refit': FULL,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable_json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def save_bytes(path, data):
    """Idempotent evidence writes: equal content is reused; changed content is refused."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing diagnostic output differs: {path}. Keep it; use a new analysis version.')
        return
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temp.write_bytes(data)
    temp.replace(path)


def save_json(path, value):
    save_bytes(path, stable_json(value).encode('utf-8'))


def save_csv(path, frame):
    save_bytes(path, frame.to_csv(index=False, lineterminator='\n').encode('utf-8'))


def read_csv(path):
    # Preserve saved floating-point values, especially scores exactly on a threshold.
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', dtype={
        'sample_id': str, 'sha256': str, 'group_id': str, 'prompt_id': str})


def align(reference, candidate, columns):
    required = {'sample_id', 'label', *columns}
    if not required.issubset(candidate):
        raise ValueError('Missing columns: ' + str(sorted(required - set(candidate))))
    if reference.sample_id.duplicated().any() or candidate.sample_id.duplicated().any():
        raise ValueError('Duplicate sample IDs')
    if set(reference.sample_id) != set(candidate.sample_id):
        raise ValueError('Prediction IDs differ from the complete manifest')
    ordered = candidate.set_index('sample_id').loc[reference.sample_id].reset_index()
    if not np.array_equal(reference.label.to_numpy(), ordered.label.to_numpy()):
        raise ValueError('Prediction labels differ from the manifest')
    if not np.isfinite(ordered[columns].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite features or scores')
    return ordered


def check_partitions(manifests, roles):
    all_rows = pd.concat(manifests.values(), ignore_index=True)
    if all_rows.sample_id.duplicated().any():
        raise ValueError('Sample ID leakage across splits')
    for col in ('sha256', 'group_id', 'local_path'):
        if (all_rows.groupby(col).split.nunique() > 1).any():
            raise ValueError(col + ' leakage across outer splits')
    generated = all_rows[all_rows.label == 1]
    prompts = generated[generated.prompt_id.astype(str).str.strip().ne('')]
    if (prompts.groupby('prompt_id').split.nunique() > 1).any():
        raise ValueError('Generated prompt leakage across splits')
    aligned = align(manifests['train'], roles, [])
    if set(aligned.role) != {'branch_train', 'calibration', 'fusion_train'}:
        raise ValueError('Expected the three original training roles')
    for col in ('sha256', 'group_id', 'prompt_id'):
        if not np.array_equal(aligned[col].to_numpy(), manifests['train'][col].to_numpy()):
            raise ValueError('Training-role metadata differs: ' + col)
        chosen = aligned if col != 'prompt_id' else aligned[
            (aligned.label == 1) & aligned.prompt_id.astype(str).str.strip().ne('')]
        if (chosen.groupby(col).role.nunique() > 1).any():
            raise ValueError(col + ' leakage between fitting roles')
    for _, group in aligned.groupby('role'):
        if set(group.label) != {0, 1}:
            raise ValueError('Each training role must contain both classes')


def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(y) != {0, 1} or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Need both classes and finite scores between zero and one')
    tn, fp, fn, tp = map(int, confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel())
    return {'n': len(y), 'auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'accuracy': (tp + tn) / len(y), 'precision': tp / max(tp + fp, 1),
            'recall': tp / (tp + fn), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'fpr': fp / (fp + tn), 'brier': float(brier_score_loss(y, scores)),
            'threshold': float(threshold), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def best_f1(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def low_fpr_threshold(y, scores, target=0.05):
    """Highest validation recall with FPR <= target; ties use the highest threshold."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    choices = [(float(((scores[y == 1] >= t).mean())), float(t)) for t in candidates
               if (scores[y == 0] >= t).mean() <= target]
    return max(choices)[1]


def load_run(root, run_name=DEFAULT_RUN):
    """Load small tables only. No checkpoint deserialization or image staging."""
    root = Path(root)
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('Use one saved run folder name')
    run = root / 'AI_Shield_Adaptation/full_pipeline_v1' / run_name
    cf_root = root / 'AI_Shield_CommunityForensics/experiments/finetune_v1'
    split_root = root / 'AI_Shield_Data/splits/v1'
    inputs = {}

    def source(path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing saved input: {path}. Mount the Drive account containing the completed run.')
        inputs[str(path.relative_to(root)).replace('\\', '/')] = sha(path)
        return path

    report = read_json(source(run / 'final_report.json'))
    if report.get('status') != 'completed':
        raise ValueError('These notebooks require the completed training report')
    plan = read_json(source(run / 'experiment_plan.json'))
    selection = read_json(source(run / 'adapted_full_selection.json'))
    lock = read_json(source(run / 'selection_lock.json'))
    historical = read_json(source(run / 'historical_reference_hashes.json'))
    if plan['feature_order'] != FULL or selection['feature_order'] != FULL:
        raise ValueError('Unexpected full-pipeline feature order')
    manifests = {}
    for split in ('train', *SPLITS):
        p = source(split_root / (split + '.csv'))
        if sha(p) != plan['split_sha256'][split]:
            raise ValueError('Changed original split: ' + split)
        manifests[split] = read_csv(p)
        if set(manifests[split].split) != {split} or set(manifests[split].label) != {0, 1}:
            raise ValueError('Bad manifest split/labels: ' + split)
    roles = read_csv(source(run / 'training_roles.csv'))
    for name in ('training_roles.csv', 'adapted_full_selection.json'):
        if sha(run / name) != lock['files'][name]:
            raise ValueError('Changed locked input: ' + name)
    check_partitions(manifests, roles)
    fusion_rows = roles[roles.role == 'fusion_train']
    train = align(fusion_rows, read_csv(source(run / 'adapted_full_fusion_training_features.csv')), FULL)
    if set(train.split) != {'train'}:
        raise ValueError('Fusion features must be training-only')
    predictions, cf = {}, {}
    threshold_path = source(cf_root / 'results/validation_thresholds.json')
    cf_threshold = float(read_json(threshold_path)['finetuned']['max_f1'])
    for split in SPLITS:
        predictions[split] = align(manifests[split], read_csv(source(
            run / f'adapted_full_{split}_predictions.csv')), FULL + ['fusion_prob'])
        cf_path = source(cf_root / f'predictions/finetuned_{split}.csv')
        cf[split] = align(manifests[split], read_csv(cf_path), ['prob_fake'])
        for path in ([cf_path, threshold_path] if split != 'validation' else []):
            suffix = str(path.relative_to(cf_root)).replace('\\', '/')
            matches = [v for k, v in historical.items() if k.replace('\\', '/').endswith('/' + suffix)]
            if len(matches) != 1 or sha(path) != matches[0]:
                raise ValueError('Changed historical CF input: ' + suffix)
        if split != 'validation':
            for name, scores, threshold in (
                ('adapted_full/fusion_prob', predictions[split].fusion_prob, selection['thresholds']['fusion_prob']),
                ('CF/finetuned', cf[split].prob_fake, cf_threshold)):
                matches = [x for x in report['comparison'] if x['model'] == name and x['split'] == split]
                measured = metrics(manifests[split].label, scores, threshold)
                if len(matches) != 1 or any(not np.isclose(measured[k], matches[0][k], atol=1e-9, rtol=1e-8)
                                          for k in measured):
                    raise ValueError('Saved comparison cannot be reproduced: ' + name + '/' + split)
    actual_val = metrics(predictions['validation'].label, predictions['validation'].fusion_prob,
                         selection['thresholds']['fusion_prob'])
    if any(not np.isclose(actual_val[k], selection['validation_metrics'][k], atol=1e-9, rtol=1e-8)
           for k in actual_val):
        raise ValueError('Saved validation result differs')
    histories = {b: read_csv(source(run / (b + '_history.csv'))) for b in ('effnet', 'semantic')}
    return dict(root=root, run=run, report=report, plan=plan, selection=selection,
                manifests=manifests, train=train, roles=roles, predictions=predictions,
                cf=cf, cf_threshold=cf_threshold, histories=histories, input_hashes=inputs)


def output_dir(data, notebook, options=None):
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'Pillow')}
    identity = {'version': VERSION, 'notebook': notebook, 'options': options or {},
                'input_hashes': data['input_hashes'], 'libraries': versions,
                'source_sha256': sha(__file__)}
    key = hashlib.sha256(stable_json(identity).encode()).hexdigest()[:16]
    folder = data['root'] / 'AI_Shield_Diagnostics' / data['run'].name / (notebook + '_' + key)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / 'analysis_inputs.json', identity)
    print('Results:', folder)
    return folder


def baseline_scores(data, split):
    p = data['predictions'][split]
    return {'forensic': p.effnet_prob.to_numpy(), 'semantic': p.semantic_vit_prob.to_numpy(),
            'saved_full': p.fusion_prob.to_numpy(), 'CF': data['cf'][split].prob_fake.to_numpy()}


def baseline_thresholds(data):
    t = data['selection']['thresholds']
    return {'forensic': t['effnet_prob'], 'semantic': t['semantic_vit_prob'],
            'saved_full': t['fusion_prob'], 'CF': data['cf_threshold']}


def baseline_table(data):
    return pd.DataFrame([{'model': m, 'split': split, **metrics(data['predictions'][split].label, s,
                         baseline_thresholds(data)[m])} for split in SPLITS
                         for m, s in baseline_scores(data, split).items()])


def score_diagnostics(data):
    distributions, reliability = [], []
    for split in SPLITS:
        y = data['predictions'][split].label.to_numpy()
        for model, scores in baseline_scores(data, split).items():
            for label in (0, 1):
                values = scores[y == label]
                q = np.quantile(values, [.05, .25, .5, .75, .95])
                distributions.append(dict(split=split, model=model, label=label, n=len(values),
                    p05=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4],
                    fraction_below_001=float((values < .01).mean()),
                    fraction_above_099=float((values > .99).mean())))
            bins = np.minimum((scores * 10).astype(int), 9)
            for b in range(10):
                mask = bins == b
                if not mask.any():
                    continue
                reliability.append(dict(split=split, model=model, bin_lower=b / 10, n=int(mask.sum()),
                    mean_score=float(scores[mask].mean()), actual_generated_fraction=float(y[mask].mean())))
    return pd.DataFrame(distributions), pd.DataFrame(reliability)


def threshold_diagnostics(data):
    """Choose alternative thresholds on validation ONLY; freeze before evaluating tests."""
    y = data['predictions']['validation'].label.to_numpy()
    chosen = {}
    for model, scores in baseline_scores(data, 'validation').items():
        chosen[model] = {'saved': baseline_thresholds(data)[model],
                         'validation_fpr_05': low_fpr_threshold(y, scores)}
    rows = []
    for split in SPLITS:
        for model, scores in baseline_scores(data, split).items():
            for operating_point, threshold in chosen[model].items():
                rows.append(dict(model=model, split=split, operating_point=operating_point,
                    **metrics(data['predictions'][split].label, scores, threshold)))
    return chosen, pd.DataFrame(rows)


def fit_ablations(data):
    """Only the reserved fusion TRAIN rows fit weights. No test access here."""
    train, validation = data['train'], data['predictions']['validation']
    if set(train.split) != {'train'} or set(validation.split) != {'validation'}:
        raise ValueError('Wrong fitting/selection split')
    expected_ids = set(data['roles'].loc[data['roles'].role == 'fusion_train', 'sample_id'])
    if set(train.sample_id) != expected_ids or set(train.sample_id) & set(validation.sample_id):
        raise ValueError('Fusion fitting IDs are not the reserved training IDs')
    models, choices, rows = {}, {}, []
    for name, columns in ABLATIONS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=5000, random_state=191))
        model.fit(train[columns], train.label)
        if int(model[-1].n_iter_.max()) >= model[-1].max_iter:
            raise RuntimeError('Fusion did not converge: ' + name)
        scores = model.predict_proba(validation[columns])[:, 1]
        threshold = best_f1(validation.label, scores)
        choices[name] = {'columns': columns, 'threshold': threshold,
                         'validation_metrics': metrics(validation.label, scores, threshold)}
        models[name] = model
        rows.append(dict(model=name, **choices[name]['validation_metrics']))
    # This control must reproduce the saved full fusion; otherwise comparisons are invalid.
    scores = models['full_refit'].predict_proba(validation[FULL])[:, 1]
    error = float(np.abs(scores - validation.fusion_prob.to_numpy()).max())
    if not np.allclose(scores, validation.fusion_prob, atol=5e-4, rtol=1e-4):
        raise ValueError(f'Full-refit control differs from saved validation predictions ({error}). Stop; do not relax the check.')
    candidates = {name: x['validation_metrics'] for name, x in choices.items()}
    for name in ('semantic', 'forensic', 'saved_full'):
        candidates[name] = metrics(validation.label, baseline_scores(data, 'validation')[name], baseline_thresholds(data)[name])
    winner = max(sorted(candidates), key=lambda n: (candidates[n]['f1'], candidates[n]['auc']))
    lock = {'selection_split': 'validation', 'rule': 'highest validation F1, then AUROC, alphabetical tie-break',
            'winner_among_AI_Shield_variants': winner, 'CF_is_reference_not_selected': True,
            'full_refit_max_validation_error': error, 'candidates': choices,
            'existing_candidates': {n: candidates[n] for n in ('semantic', 'forensic', 'saved_full')},
            'fit_sample_ids': sorted(train.sample_id.tolist()), 'C': 1., 'seed': 191,
            'test_policy': 'Historical exploratory comparison; fresh confirmation required before deployment.'}
    return models, lock, pd.DataFrame(rows)


def evaluate_ablations(data, models, lock, folder):
    # Require the pre-test choice to have been written and read back.
    if read_json(Path(folder) / 'validation_selection.json') != lock:
        raise ValueError('Save the validation selection before evaluating tests')
    rows, predictions = [], {}
    for split in SPLITS:
        frame = data['predictions'][split]
        scores = baseline_scores(data, split)
        thresholds = baseline_thresholds(data)
        for name, model in models.items():
            columns = lock['candidates'][name]['columns']
            scores[name] = model.predict_proba(frame[columns])[:, 1]
            thresholds[name] = lock['candidates'][name]['threshold']
        result = frame[['sample_id', 'label', 'split']].copy()
        for name, values in scores.items():
            result[name] = values
            rows.append(dict(model=name, split=split, **metrics(frame.label, values, thresholds[name])))
        predictions[split] = result
        save_csv(Path(folder) / (split + '_scores.csv'), result)
    table = pd.DataFrame(rows)
    save_csv(Path(folder) / 'ablation_comparison.csv', table)
    return table, predictions


def coefficient_table(models):
    rows = []
    for name, model in models.items():
        for feature, weight in zip(ABLATIONS[name], model[-1].coef_[0]):
            rows.append(dict(model=name, feature=feature, standardized_weight=float(weight)))
    return pd.DataFrame(rows)


def error_table(data, split):
    meta = data['manifests'][split]
    columns = ['sample_id', 'label', 'split', 'generator', 'local_path', 'sha256']
    columns += [c for c in ('source_dataset', 'source', 'category', 'group_id', 'prompt_id') if c in meta]
    result = meta[columns].copy()
    y = result.label.to_numpy()
    for name, scores in baseline_scores(data, split).items():
        result[name + '_score'] = scores
        result[name + '_prediction'] = (scores >= baseline_thresholds(data)[name]).astype(int)
        result[name + '_correct'] = result[name + '_prediction'].to_numpy() == y
    full, cf = result.saved_full_correct, result.CF_correct
    result['case'] = np.select([full & cf, ~full & cf, full & ~cf],
        ['both_correct', 'CF_only_correct', 'full_only_correct'], default='both_wrong')
    result['original_extension'] = result.local_path.map(lambda p: Path(p).suffix.lower())
    return result


def subgroup_table(errors):
    """Class-conditional rates; never describe real-only accuracy as AI recall."""
    rows = []
    grouping = ['generator', 'original_extension']
    grouping += [c for c in ('source_dataset', 'source', 'category') if c in errors]
    for split, outer in errors.groupby('split'):
        for field in grouping:
            for (group, label), frame in outer.groupby([field, 'label'], dropna=False):
                for model in ('forensic', 'semantic', 'saved_full', 'CF'):
                    positive = float(frame[model + '_prediction'].mean())
                    rows.append(dict(split=split, field=field, group=str(group), label=int(label), n=len(frame),
                        model=model, metric='generated_recall' if label == 1 else 'real_false_positive_rate',
                        value=positive, small_group=len(frame) < 30))
    return pd.DataFrame(rows)


def select_gallery(errors, per_case=3):
    # Deterministic samples from both labels; not a representative prevalence estimate.
    wanted = errors[errors.case != 'both_correct'].sort_values('sample_id')
    return wanted.groupby(['case', 'label'], group_keys=False).head(per_case).reset_index(drop=True)


def gallery_path(original, root):
    text = str(original).replace('\\', '/')
    marker = '/content/drive/MyDrive/'
    return Path(root) / text[len(marker):] if text.startswith(marker) else Path(original)


def render_gallery(selected, root):
    """Read only selected originals. Missing images yield a labelled placeholder."""
    from PIL import Image, ImageDraw, ImageOps
    n = len(selected)
    if not n:
        return None, pd.DataFrame(columns=['sample_id', 'status'])
    width, height, cols = 330, 300, 3
    canvas = Image.new('RGB', (cols * width, ((n + cols - 1) // cols) * height), 'white')
    draw = ImageDraw.Draw(canvas)
    checks = []
    for i, row in enumerate(selected.itertuples()):
        x, y = (i % cols) * width, (i // cols) * height
        path = gallery_path(row.local_path, root)
        status = 'missing_image'
        if path.is_file():
            if sha(path) != row.sha256:
                status = 'hash_mismatch_not_displayed'
            else:
                try:
                    with Image.open(path) as im:
                        im.seek(0)
                        tile = ImageOps.contain(im.convert('RGB'), (width - 16, 210))
                    canvas.paste(tile, (x + 8, y + 4))
                    status = 'verified'
                except (OSError, ValueError):
                    status = 'unreadable_image'
        if status != 'verified':
            draw.text((x + 8, y + 85), status, fill='red')
        caption = (f'{row.sample_id}\n{row.case} | label={row.label}\n'
                   f'CF={row.CF_score:.3f} full={row.saved_full_score:.3f}\n{row.generator}')
        draw.text((x + 8, y + 215), caption, fill='black')
        checks.append(dict(sample_id=row.sample_id, status=status))
    return canvas, pd.DataFrame(checks)


def interpretation(data, ablations=None):
    """Evidence-led observations, not an automatic choice of a new deployment model."""
    base = baseline_table(data)
    test = base[base.split == 'test'].set_index('model')
    messages = [
        f"Main test: forensic {test.loc['forensic','accuracy']:.1%}, semantic {test.loc['semantic','accuracy']:.1%}, full {test.loc['saved_full','accuracy']:.1%}, CF {test.loc['CF','accuracy']:.1%} accuracy.",
        'AUROC compares score ranking across thresholds; low AUROC cannot be repaired by threshold changes alone.',
        'A large gap between real and generated score distributions is desirable. Extreme scores are not proof of calibrated confidence.',
        'These historical tests have already informed research. Any new winner needs a fresh, untouched confirmation set.',
        'This analysis does not test JPEG/resize robustness, new generators, EXIF/C2PA, or historical/context reasoning.',
        'Do not deploy a new threshold or fusion simply because one historical test number increased.',
    ]
    if ablations is not None:
        val = ablations[ablations.split == 'validation'].sort_values(['f1', 'auc'], ascending=False)
        messages.append('Validation ranking is saved separately from historical test performance. Inspect false positives as well as F1.')
        messages.append('Highest validation F1 in the displayed table (including CF reference): ' + str(val.iloc[0]['model']))
    return messages
