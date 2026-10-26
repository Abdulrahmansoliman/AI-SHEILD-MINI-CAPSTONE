"""Full 17-feature adaptation; original artifacts and app remain read-only."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ai_shield_adaptation import (
    Config, FEATURES, SPLIT_HASHES, align_predictions, checked_csv, max_f1_threshold,
    open_rgb, partition_training, predict_pair, preflight, scored_metrics,
    sha256_file, write_json,
)

AUX_FEATURES = ['vae_recon_error', 'vae_recon_percentile', 'face_present',
    'num_faces', 'largest_face_ratio', 'face_confidence',
    'openforensics_applicable', 'openforensics_prob_fake']
FULL_FEATURES = FEATURES[:2] + ['effnet_calibrated_prob'] + AUX_FEATURES + FEATURES[2:]


@dataclass(frozen=True)
class FullConfig(Config):
    # About 70% branch / 15% calibration / 15% fusion, grouped before fitting.
    fusion_fraction: float = 0.30

    @property
    def extras(self):
        phase1 = self.root / 'AI_Shield_Phase1'
        phase3 = self.root / 'AI_Shield_Phase3_Fusion/forensic_semantic_glm_v1'
        return {
            'vae': phase1 / 'vae_real_only_tinygenimage_v1/vae_real_only_tinygenimage_v1_best.pt',
            'vae_reference_errors': phase1 / 'vae_real_only_tinygenimage_v1/vae_reconstruction_errors.npz',
            'platt_calibrator': phase1 / 'effnet_b0_tinygenimage_full_balanced_v1/platt_calibrator.pkl',
            'openforensics_h5': phase1 / 'openforensics_effnetv2b0/dffnetv2B0.h5',
            'openforensics_json': phase1 / 'openforensics_effnetv2b0/dffnetv2B0.json',
            'full_glm': phase3 / 'checkpoints/phase3_full_forensic_semantic_glm_v1.pkl',
            'phase3_metrics': phase3 / 'results/phase3_fusion_metrics_v1.json',
        }


def full_preflight(config):
    frames, audit = preflight(config)
    for key, path in config.extras.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing original {key}: {path}')
        expected = audit['source_artifacts'].get(key + '_sha256')
        if expected and sha256_file(path) != expected:
            raise ValueError(f'Original {key} differs from the completed audit')
    for split in ('test', 'unseen_generator'):
        saved = checked_csv(config.audit_root / f'aishield_{split}_predictions.csv')
        align_predictions(frames[split], saved, FULL_FEATURES + ['aishield_full_prob'])
    return frames, audit


def full_partition(train, config):
    partition = partition_training(train, config.fusion_fraction, config.seed)
    reserve = partition[partition.role == 'fusion_train'].copy()
    divided = partition_training(reserve, 0.5, config.seed + 1)
    roles = divided.set_index('sample_id').role.replace({'branch_train': 'calibration'})
    selected = partition.sample_id.isin(roles.index)
    partition.loc[selected, 'role'] = partition.loc[selected, 'sample_id'].map(roles)
    if set(partition.role) != {'branch_train', 'calibration', 'fusion_train'}:
        raise ValueError('Three independent training roles are required')
    return partition


def prepare_full_run(config, frames):
    if config.head_epochs < 1 or config.partial_epochs < 0 or config.batch_size < 1:
        raise ValueError('Invalid training budget')
    partition = full_partition(frames['train'], config)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    run = config.root / 'AI_Shield_Adaptation/full_pipeline_v1' / stamp
    run.mkdir(parents=True, exist_ok=False)
    partition.to_csv(run / 'training_roles.csv', index=False)
    inputs = {**config.checkpoints, **config.extras}
    for split in ('test', 'unseen_generator'):
        inputs['audit_' + split] = config.audit_root / f'aishield_{split}_predictions.csv'
    write_json(run / 'input_hashes.json', {k: {'path': str(p), 'sha256': sha256_file(p)} for k, p in inputs.items()})
    write_json(run / 'experiment_plan.json', {
        'config': asdict(config), 'feature_order': FULL_FEATURES, 'split_sha256': SPLIT_HASHES,
        'role_counts': partition.groupby(['role', 'label']).size().rename('n').reset_index().to_dict('records'),
        'libraries': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'timm', 'transformers', 'scikit-learn', 'numpy', 'pandas', 'Pillow')},
        'scope': 'Original full 17-feature architecture; adapt EfficientNet and semantic head/backbone, refit calibration and full fusion; VAE and face model frozen.',
        'budget': 'Same overall 2990-image training pool as CF, fewer branch-training images; compute and pretraining are not matched.',
        'vae_protocol': 'Preserve sampled-latent VAE with one reproducible CPU sample seeded by image hash on every split and in live inference. Original historical results retain their original cached draws as a separate reference.',
        'test_policy': 'Previously inspected historical tests; no training, calibration, threshold or checkpoint selection on them.',
    })
    return run, partition


def verify_input_lock(config, run):
    for key, item in json.loads((Path(run) / 'input_hashes.json').read_text()).items():
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError(f'Original input changed during the run: {key}')


def stage_images(frames, destination):
    """Copy verified bytes to runtime disk, avoiding repeated Drive I/O in epochs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    total = sum(Path(r.local_path).stat().st_size for f in frames.values() for r in f.itertuples())
    if shutil.disk_usage(destination).free < total + 1024 ** 3:
        raise RuntimeError('Not enough runtime disk; set STAGE_TO_LOCAL=False in setup')
    staged = {}
    from tqdm.auto import tqdm
    for split, frame in frames.items():
        copy = frame.copy()
        paths = []
        for row in tqdm(frame.itertuples(), total=len(frame), desc='Stage ' + split):
            target = destination / (row.sample_id + Path(row.local_path).suffix)
            shutil.copyfile(row.local_path, target)
            if sha256_file(target) != row.sha256:
                raise ValueError('Copied bytes changed: ' + row.sample_id)
            paths.append(str(target))
        copy['local_path'] = paths
        staged[split] = copy
    return staged


def load_legacy_fusion(path):
    # This is an owned artifact whose hash is checked against the previous audit.
    model = joblib.load(path)
    for _, step in getattr(model, 'steps', []):
        if step.__class__.__name__ == 'SimpleImputer' and not hasattr(step, '_fill_dtype'):
            step._fill_dtype = np.asarray(step.statistics_).dtype
    if list(model.feature_names_in_) != FULL_FEATURES:
        raise ValueError('Saved fusion feature order differs from the 17-feature schema')
    return model


def replay_original(config, frames, run):
    model = load_legacy_fusion(config.extras['full_glm'])
    checks = []
    for split in ('test', 'unseen_generator'):
        saved = align_predictions(frames[split], checked_csv(config.audit_root / f'aishield_{split}_predictions.csv'), FULL_FEATURES + ['aishield_full_prob'])
        actual = model.predict_proba(saved[FULL_FEATURES])[:, 1]
        if not np.allclose(actual, saved.aishield_full_prob, atol=1e-7, rtol=1e-6):
            raise ValueError('Original full fusion does not reproduce saved scores: ' + split)
        checks.append({'split': split, 'n': len(saved), 'max_absolute_error': float(np.max(np.abs(actual - saved.aishield_full_prob)))})
    write_json(Path(run) / 'original_full_fusion_replay.json', checks)
    return checks


def latent_seed(identity):
    return int(hashlib.sha256(('vae-4242:' + str(identity)).encode()).hexdigest()[:8], 16)


class FrozenAuxiliary:
    """Original sampled VAE + Haar face gate + original OpenForensics, all frozen."""
    def __init__(self, config):
        import torch
        from torch import nn
        from torchvision import transforms
        import cv2

        class ConvVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.ReLU(True),nn.Conv2d(32,64,4,2,1),nn.ReLU(True),nn.Conv2d(64,128,4,2,1),nn.ReLU(True),nn.Conv2d(128,256,4,2,1),nn.ReLU(True))
                self.fc_mu = nn.Linear(16384,128)
                self.fc_logvar = nn.Linear(16384,128)
                self.fc_decode = nn.Linear(128,16384)
                self.decoder = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU(True),nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(True),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(True),nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())
            def encode(self, x):
                h = self.encoder(x).view(x.shape[0], -1)
                return self.fc_mu(h), self.fc_logvar(h)
            def decode(self, z):
                return self.decoder(self.fc_decode(z).view(z.shape[0],256,8,8))

        self.vae = ConvVAE()
        self.vae.load_state_dict(torch.load(config.extras['vae'], map_location='cpu', weights_only=False)['model_state'], strict=True)
        self.vae.eval().requires_grad_(False)
        self.vae_transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
        with np.load(config.extras['vae_reference_errors'], allow_pickle=False) as saved:
            self.reference = np.sort(np.asarray(saved['full'], dtype=float).reshape(-1))
        if not len(self.reference) or not np.isfinite(self.reference).all():
            raise ValueError('Invalid original VAE reference distribution')
        cascade = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.face = cv2.CascadeClassifier(str(cascade))
        if self.face.empty():
            raise RuntimeError('Cannot load the original Haar face gate')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        self.tf = tf
        with tf.device('/CPU:0'):
            try:
                self.face_model = tf.keras.models.model_from_json(config.extras['openforensics_json'].read_text())
                self.face_model.load_weights(str(config.extras['openforensics_h5']))
            except (ValueError, TypeError, KeyError):
                # Same legacy reconstruction used successfully in the saved audit.
                self.face_model = tf.keras.Sequential([
                    tf.keras.applications.EfficientNetV2B0(weights=None, input_shape=(256,256,3), include_top=False, pooling='max', include_preprocessing=True),
                    tf.keras.layers.Dense(1, activation='sigmoid', name='dense_2')], name='dffnetv2b0')
                self.face_model(tf.zeros((1,256,256,3)), training=False)
                self.face_model.load_weights(str(config.extras['openforensics_h5']))
        self.face_model.trainable = False

    def face_features(self, image):
        import cv2
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        try:
            rects, _, weights = self.face.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40), outputRejectLevels=True)
            weights = list(weights) if len(rects) else []
        except cv2.error:
            rects = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
            weights = [1.0] * len(rects)
        valid = [(float(w*h/(image.width*image.height)), float(c)) for (_,_,w,h),c in zip(rects,weights) if c >= 0 and w*h/(image.width*image.height) >= .01]
        score = .5
        if valid:
            batch = np.asarray(image.resize((256,256)), dtype=np.float32)[None]
            with self.tf.device('/CPU:0'):
                p = np.asarray(self.face_model(batch, training=False)).reshape(-1)
            if len(p) != 1 or not np.isfinite(p).all() or not 0 <= p[0] <= 1:
                raise ValueError('Invalid OpenForensics output')
            score = 1.0 - float(p[0])
        return {'face_present': int(bool(valid)), 'num_faces': len(valid),
            'largest_face_ratio': max([v[0] for v in valid], default=0.),
            'face_confidence': max([v[1] for v in valid], default=0.),
            'openforensics_applicable': int(bool(valid)), 'openforensics_prob_fake': score}

    def features(self, image, identity):
        import torch
        with torch.inference_mode():
            x = self.vae_transform(image)[None]
            mu, logvar = self.vae.encode(x)
            rng = torch.Generator(device='cpu').manual_seed(latent_seed(identity))
            eps = torch.randn(mu.shape, generator=rng, dtype=mu.dtype)
            reconstruction = self.vae.decode(mu + eps * torch.exp(.5 * logvar))
            error = float((x - reconstruction).square().mean())
        if not np.isfinite(error):
            raise ValueError('Non-finite VAE reconstruction')
        return {'vae_recon_error': error, 'vae_recon_percentile': float(np.searchsorted(self.reference,error,side='right')/len(self.reference)), **self.face_features(image)}


def verify_face_parity(aux, config, frames, run):
    saved = checked_csv(config.audit_root / 'aishield_test_predictions.csv')
    selected = pd.concat([saved[saved.face_present == 1].head(3), saved[saved.face_present == 0].head(3)])
    if set(selected.face_present) != {0,1}:
        raise ValueError('Need both face and no-face reference cases')
    paths = frames['test'].set_index('sample_id').local_path
    errors = []
    for r in selected.itertuples():
        actual = aux.face_features(open_rgb(paths.loc[r.sample_id]))
        for key, value in actual.items():
            expected = getattr(r, key)
            if not np.isclose(value, expected, atol=5e-4, rtol=1e-4):
                raise ValueError(f'Face parity failed: {r.sample_id} / {key}')
        errors.append({'sample_id': r.sample_id, 'status': 'pass'})
    write_json(Path(run) / 'face_parity.json', errors)
    return errors


def cache_auxiliary(aux, frames, config, run):
    from tqdm.auto import tqdm
    tables = {}
    for split, frame in frames.items():
        rows = []
        for r in tqdm(frame.itertuples(), total=len(frame), desc='Frozen auxiliary: '+split):
            rows.append({'sample_id':r.sample_id,'label':r.label, **aux.features(open_rgb(r.local_path), r.sha256)})
        result = align_predictions(frame, pd.DataFrame(rows), AUX_FEATURES)
        result['split'] = split
        result.to_csv(Path(run) / f'auxiliary_{split}.csv', index=False)
        tables[split] = result
    write_json(Path(run) / 'auxiliary_cache_hashes.json', {s:sha256_file(Path(run)/f'auxiliary_{s}.csv') for s in tables})
    return tables


def combine_features(main, aux, calibrator):
    chosen = aux[aux.sample_id.isin(main.sample_id)]
    ordered = align_predictions(main, chosen, AUX_FEATURES)
    result = main.copy()
    for key in AUX_FEATURES:
        result[key] = ordered[key].to_numpy()
    result['effnet_calibrated_prob'] = calibrator.predict_proba(result[['effnet_logit']])[:,1]
    align_predictions(result, result, FULL_FEATURES)
    return result


def fit_full_models(calibration, fusion_main, aux, branch_ids):
    if set(calibration.split) != {'train'} or set(fusion_main.split) != {'train'}:
        raise ValueError('Calibration and fusion fitting must use training data only')
    sets = [set(calibration.sample_id),set(fusion_main.sample_id),set(branch_ids)]
    if any(sets[a] & sets[b] for a,b in [(0,1),(0,2),(1,2)]):
        raise ValueError('Branch, calibration and fusion images must be disjoint')
    for frame in (calibration, fusion_main):
        align_predictions(frame, frame, FEATURES)
        if set(frame.label) != {0,1}:
            raise ValueError('Each fitting role needs both classes')
    calibrator = LogisticRegression(C=1.,max_iter=2000,random_state=191)
    calibrator.fit(calibration[['effnet_logit']],calibration.label)
    full = combine_features(fusion_main, aux, calibrator)
    fusion = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.,max_iter=5000,random_state=191))
    fusion.fit(full[FULL_FEATURES],full.label)
    return calibrator, fusion, full


def fit_full_variant(name, models, processor, partition, frames, auxiliary, config, run, device):
    development = partition[partition.role != 'branch_train']
    main = predict_pair(models,processor,development,config,device)
    roles = partition.set_index('sample_id').role
    calibration = main[main.sample_id.map(roles) == 'calibration']
    fusion_main = main[main.sample_id.map(roles) == 'fusion_train']
    start = time.perf_counter()
    calibrator, fusion, full_train = fit_full_models(calibration,fusion_main,auxiliary['train'],partition.loc[partition.role=='branch_train','sample_id'])
    fit_seconds = time.perf_counter()-start
    val = combine_features(predict_pair(models,processor,frames['validation'],config,device),auxiliary['validation'],calibrator)
    val['fusion_prob'] = fusion.predict_proba(val[FULL_FEATURES])[:,1]
    thresholds = {k:max_f1_threshold(val.label,val[k]) for k in ['effnet_prob','effnet_calibrated_prob','semantic_vit_prob','fusion_prob']}
    joblib.dump(calibrator,Path(run)/f'{name}_calibrator.joblib')
    joblib.dump(fusion,Path(run)/f'{name}_fusion.joblib')
    calibration.to_csv(Path(run)/f'{name}_calibration_predictions.csv',index=False)
    full_train.to_csv(Path(run)/f'{name}_fusion_training_features.csv',index=False)
    val.to_csv(Path(run)/f'{name}_validation_predictions.csv',index=False)
    selection = {'thresholds':thresholds,'feature_order':FULL_FEATURES,'selection_split':'validation',
        'fusion_training_n':len(full_train),'calibration_n':len(calibration),'fit_seconds':fit_seconds,
        'validation_metrics':scored_metrics(val.label,val.fusion_prob,thresholds['fusion_prob'])}
    write_json(Path(run)/f'{name}_selection.json',selection)
    return calibrator,fusion,thresholds,selection


def evaluate_full_variant(name, models, processor, frames, auxiliary, calibrator, fusion, thresholds, config, run, device):
    metrics, subgroups = [], []
    for split in ('test','unseen_generator'):
        pred = combine_features(predict_pair(models,processor,frames[split],config,device),auxiliary[split],calibrator)
        pred['fusion_prob'] = fusion.predict_proba(pred[FULL_FEATURES])[:,1]
        pred['generator'] = frames[split].generator.to_numpy()
        pred.to_csv(Path(run)/f'{name}_{split}_predictions.csv',index=False)
        for key,threshold in thresholds.items():
            metrics.append({'model':name+'/'+key,'split':split,**scored_metrics(pred.label,pred[key],threshold)})
        for generator,g in pred.groupby('generator'):
            positives = (g.fusion_prob >= thresholds['fusion_prob']).sum()
            subgroups.append({'model':name,'split':split,'generator':generator,'n':len(g),
                'metric':'generated_recall' if g.label.eq(1).all() else 'real_false_positive_rate',
                'value':float(positives/len(g))})
    pd.DataFrame(subgroups).to_csv(Path(run)/f'{name}_generator_breakdown.csv',index=False)
    return metrics,subgroups


def lock_full_selection(run, selections):
    run = Path(run)
    names = ['effnet_adapted.pt','semantic_adapted.pt','input_hashes.json','training_roles.csv','auxiliary_cache_hashes.json']
    names += [f'auxiliary_{s}.csv' for s in ('train','validation','test','unseen_generator')]
    for variant in selections:
        names += [f'{variant}_{suffix}' for suffix in ('calibrator.joblib','fusion.joblib','selection.json')]
    winner = max(sorted(selections),key=lambda v:(selections[v]['validation_metrics']['f1'],selections[v]['validation_metrics']['auc']))
    lock = {'files':{n:sha256_file(run/n) for n in names},'validation_selected_candidate':winner,
        'rule':'Highest validation F1, then validation AUROC; never select using test results.'}
    write_json(run/'selection_lock.json',lock)
    return lock


def verify_full_lock(run):
    lock = json.loads((Path(run)/'selection_lock.json').read_text())
    for name,digest in lock['files'].items():
        if sha256_file(Path(run)/name) != digest:
            raise ValueError('Locked experiment file changed: '+name)
    return lock


class FullPredictor:
    def __init__(self,models,processor,auxiliary,calibrator,fusion,device,semantic_batch_size=16):
        from src.ai_shield_adaptation import PairPredictor
        if isinstance(semantic_batch_size, bool) or not isinstance(semantic_batch_size, int) or semantic_batch_size < 1:
            raise ValueError('semantic_batch_size must be a positive integer')
        self.semantic_batch_size = semantic_batch_size
        self.pair = PairPredictor(models,processor,fusion,device)
        self.auxiliary,self.calibrator,self.fusion = auxiliary,calibrator,fusion

    def predict(self,path,identity=None):
        import torch
        pair = self.pair
        image = open_rgb(path)
        with torch.inference_mode():
            logit = pair.models['effnet'](pair.transform(image)[None].to(pair.device))[0]
            # Match evaluation's physical batch size; keep only the first result.
            # cuDNN/ViT numerical paths can differ for a physical batch of one.
            pixels = pair.processor(images=image,return_tensors='pt')['pixel_values']
            if pixels.shape[0] != 1:
                raise ValueError('Standalone inference expects one image')
            pixels = pixels.repeat(self.semantic_batch_size, 1, 1, 1).to(pair.device)
            sem = pair.models['semantic'](pixels)
            row = {'effnet_logit':float(logit),'effnet_prob':float(torch.sigmoid(logit)),
                'semantic_vit_logit':float(sem['logits'][0]),'semantic_vit_prob':float(torch.sigmoid(sem['logits'][0]))}
            for key in ('attention_entropy','attention_focus','self_attention_entropy','self_attention_focus'):
                row['semantic_'+key] = float(sem[key][0])
        row.update(self.auxiliary.features(image,identity or sha256_file(path)))
        row['effnet_calibrated_prob'] = float(self.calibrator.predict_proba(pd.DataFrame([row])[['effnet_logit']])[0,1])
        values = pd.DataFrame([row])[FULL_FEATURES]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('Non-finite full-pipeline features')
        return float(self.fusion.predict_proba(values)[0,1])


def load_full_predictor(config,run,device='cpu',variant='adapted_full'):
    import torch
    from src.ai_shield_adaptation_models import load_original_models
    verify_full_lock(run)
    verify_input_lock(config,run)
    if variant not in ('adapted_full','frozen_full'):
        raise ValueError('Unknown full-pipeline variant')
    models,processor = load_original_models(config)
    if variant == 'adapted_full':
        for branch,model in models.items():
            model.load_state_dict(torch.load(Path(run)/f'{branch}_adapted.pt',map_location='cpu',weights_only=True)['model_state'],strict=True)
    saved_batch_size = int(json.loads((Path(run)/'experiment_plan.json').read_text())['config']['batch_size'])
    return FullPredictor(models,processor,FrozenAuxiliary(config),joblib.load(Path(run)/f'{variant}_calibrator.joblib'),joblib.load(Path(run)/f'{variant}_fusion.joblib'),device,semantic_batch_size=saved_batch_size)
