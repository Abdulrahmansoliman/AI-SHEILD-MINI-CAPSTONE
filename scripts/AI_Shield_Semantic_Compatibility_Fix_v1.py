"""Apply to the existing Colab session; never restart, train, or relax parity.

Run with exec(Path(...).read_text()) after CONFIG, frames, RUN, DEVICE exist.
Changes are saved only after the original eight-feature GPU parity check passes.
"""
from pathlib import Path
import hashlib
import importlib
import json
import types

OLD_FORWARD = '''    def forward(self, pixel_values):
        output = self.vit(pixel_values=pixel_values, return_dict=True)
        return self.semantic_head(output.last_hidden_state[:, 1:, :])'''
NEW_FORWARD = '''    def forward(self, pixel_values):
        # The saved audit used cuDNN's default TF32 convolution arithmetic.
        # Scope it to ViT so CF and the other branches retain their own settings.
        precision = torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=True,
        ) if pixel_values.device.type == "cuda" else nullcontext()
        with precision:
            output = self.vit(pixel_values=pixel_values, return_dict=True)
        return self.semantic_head(output.last_hidden_state[:, 1:, :])'''
OLD_PROCESSOR = 'AutoImageProcessor.from_pretrained(SEMANTIC_ID, revision=SEMANTIC_REVISION, use_fast=False)'
NEW_PROCESSOR = OLD_PROCESSOR.replace('use_fast=False', 'use_fast=True')


def patched_source(source):
    for old, new in ((OLD_FORWARD, NEW_FORWARD), (OLD_PROCESSOR, NEW_PROCESSOR)):
        if new in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError('Unrecognized source: no files changed. Share this error before continuing.')
        source = source.replace(old, new, 1)
    if 'from contextlib import nullcontext\n' not in source:
        if source.count('import math\n') != 1:
            raise RuntimeError('Unexpected imports: no files changed.')
        source = source.replace('import math\n', 'import math\nfrom contextlib import nullcontext\n', 1)
    compile(source, 'semantic_compatibility_candidate', 'exec')
    return source


def apply_fix(namespace):
    import torch
    import src.ai_shield_adaptation_models as current
    from src.ai_shield_adaptation import check_audit_parity
    required = ('CONFIG', 'frames', 'RUN', 'DEVICE')
    missing = [key for key in required if key not in namespace]
    if missing:
        raise RuntimeError('Run in the existing stopped session; missing: ' + ', '.join(missing))
    run = Path(namespace['RUN'])
    if not run.is_dir():
        raise RuntimeError('The existing RUN folder is missing; no files changed.')
    if any((run / name).exists() for name in ('effnet_adapted.pt', 'semantic_adapted.pt', 'selection_lock.json')):
        raise RuntimeError('Training artifacts already exist. Refusing to change their implementation.')
    if str(namespace['DEVICE']).split(':')[0] != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This compatibility check requires the current Colab GPU session.')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('This GPU lacks TF32. The saved audit used A100 arithmetic; use an A100 or L4 runtime. No files changed.')

    paths = [Path(current.__file__).resolve(), run / 'code/src/ai_shield_adaptation_models.py']
    plans = [(p, p.read_bytes(), patched_source(p.read_text(encoding='utf-8'))) for p in paths]
    candidate = types.ModuleType('semantic_compatibility_candidate')
    exec(compile(plans[0][2], str(paths[0]), 'exec'), candidate.__dict__)
    new_models, new_processor = candidate.load_original_models(namespace['CONFIG'])
    print('Checking the original tolerance on', torch.cuda.get_device_name(), '...', flush=True)
    report = check_audit_parity(new_models, new_processor, namespace['frames'], namespace['CONFIG'], namespace['DEVICE'])

    # Nothing on disk is changed unless every original feature passes above.
    changes = []
    for path, before, after in plans:
        after_bytes = after.encode('utf-8')
        before_hash = hashlib.sha256(before).hexdigest()
        after_hash = hashlib.sha256(after_bytes).hexdigest()
        if before != after_bytes:
            backup = path.with_name(path.name + '.before-semantic-fix-' + before_hash[:12])
            if backup.exists() and backup.read_bytes() != before:
                raise RuntimeError('Backup conflict: ' + str(backup))
            if not backup.exists():
                backup.write_bytes(before)
            temporary = path.with_name(path.name + '.semantic-fix-tmp')
            temporary.write_bytes(after_bytes)
            temporary.replace(path)
        changes.append({'path': str(path), 'before_sha256': before_hash, 'after_sha256': after_hash})

    ledger = {'fix': 'fast_processor_and_scoped_cudnn_tf32', 'gpu': torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'processor': type(new_processor).__name__,
              'atol': 5e-4, 'rtol': 1e-4, 'parity': report, 'files': changes,
              'note': 'Original embedded_source_hashes.json retained; this records the later verified patch.'}
    ledger_bytes = json.dumps(ledger, indent=2).encode('utf-8')
    ledger_path = run / ('semantic_compatibility_fix_' + hashlib.sha256(ledger_bytes).hexdigest()[:12] + '.json')
    if not ledger_path.exists():
        ledger_path.write_bytes(ledger_bytes)
    importlib.invalidate_caches()
    importlib.reload(current)
    namespace.update(models=new_models, processor=new_processor, parity=report,
                     load_original_models=current.load_original_models)
    print('PASS:', report['n'], 'images; all eight original feature checks passed.')
    print('Maximum semantic logit difference:', report['max_absolute_errors']['semantic_vit_logit'])
    print('Backups and fix record saved. Checkpoints, images, and existing results were not changed.')
    if namespace.get('_AI_SHIELD_INTEGRATED'):
        print('Compatibility verified. Continuing with the remaining reference checks.')
    else:
        print('Now rerun only the original failed reference-check cell, then continue below it.')


if __name__ == '__main__':
    apply_fix(globals())
