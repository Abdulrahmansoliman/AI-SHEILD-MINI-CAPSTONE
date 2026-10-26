"""Build the self-contained, full saved-pipeline Colab experiment."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NAME = 'AI_Shield_Full_Pipeline_FineTuning_v1.ipynb'
cells = []


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    entry = {'cell_type': kind, 'id': f'full-adapt-{len(cells):02d}',
             'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        compile(source, entry['id'], 'exec')
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell('markdown', '''
    # AI Shield — fine-tune the original full pipeline
    **Integrated fixes v2 — semantic compatibility and setup recovery are built in.**
    **Goal:** find out whether your original forensic + semantic system improves after adaptation
    to the same Dataset v1 used for Community Forensics (CF).

    | Component | What this experiment does |
    |---|---|
    | EfficientNet-B0 forensic model | Start from your saved weights; fine-tune its head, then its final block |
    | ViT + your semantic attention head | Start from your saved head and original backbone; fine-tune the head, then the final block |
    | VAE reconstruction + face/OpenForensics signals | Keep the weights frozen; calculate these signals once |
    | EfficientNet calibration | Refit the score conversion using a separate training subset |
    | Full fusion | Refit the original **17-feature** combination, including auxiliary signals |
    | Community Forensics | Keep the existing detector fixed as a comparison |

    This extends your saved system, including your custom components and pretrained backbones.
    It does **not** train every model from scratch. EXIF, C2PA and the planned context layer are outside this experiment.

    We compare **original saved full pipeline → frozen networks with refitted fusion → fine-tuned full pipeline**,
    plus the original and fine-tuned CF results. Accuracy improvement is a question to test, not a promised result.

    **To run:** choose a fresh Colab **GPU runtime (Python 3.11–3.13)**, then run cells in order.
    The Drive mount will ask you to authorize access. Use a fresh session if packages have already been imported.
    This notebook has been generated and checked locally; the full Drive/GPU experiment has not run yet.
    Original checkpoints and the application detector are read-only. Section 3 reuses this session's
    experiment or restores saved setup; it creates a folder only when no existing experiment is selected.
    **Existing session:** run section 2, then 3, then 4. Do not run the old diagnostic cells at the bottom.
    **New runtime:** install section 1 first, follow its restart instruction if shown, then run from section 2.
    Setup recovery is included. Resuming an interrupted training optimizer/epoch is not yet implemented.
''')
cell('markdown', '''
    ## 1. Install the experiment's dependencies
    Uses Colab's compatible PyTorch/torchvision pair. TensorFlow runs the frozen face model on CPU.
    Supports the current Python 3.13 runtime using TensorFlow 2.20. The previous 2.19 pin did not.
    First resolves compatible binary packages without changing the runtime; only then installs them.
    Only this temporary runtime's overlapping TensorFlow/OpenCV installations are replaced.
    Unused JAX is removed here to avoid conflicting TensorFlow/XLA libraries.
    If Colab asks for a restart, restart once and continue from section 2 before importing these packages.
''')
packages = [p for p in (ROOT / 'requirements-adaptation.txt').read_text().splitlines()
            if p.strip() and not p.startswith('#')]
packages += ['tensorflow-cpu==2.20.0', 'keras==3.10.0', 'opencv-python-headless==4.11.0.86',
             'protobuf>=5.28.0,<7']
cell('code', 'PACKAGES = ' + repr(packages) + '\n' + textwrap.dedent('''
    import subprocess, sys, os, json, importlib.metadata
    from pathlib import Path
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError('This setup supports Python 3.11, 3.12 and 3.13. No packages were changed.')
    print('Python:', sys.version.split()[0], flush=True)
    loaded_before = [name for name in ('numpy', 'pandas', 'sklearn', 'PIL', 'tensorflow', 'keras', 'cv2')
                     if name in sys.modules]
    # Keep Colab's CUDA-enabled torch/torchvision pair exactly as installed.
    constraints = Path('/content/ai_shield_torch_constraints.txt')
    constraints.write_text(''.join(name + '==' + importlib.metadata.version(name) + chr(10)
                                   for name in ('torch', 'torchvision')))
    install_options = ['--only-binary=:all:', '--constraint', str(constraints)]
    log_path = Path('/content/ai_shield_setup.log')
    log_path.write_text('Python ' + sys.version + chr(10))
    def run_pip(arguments, stage):
        print(stage + '...', flush=True)
        result = subprocess.run([sys.executable, '-m', 'pip', *arguments],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with log_path.open('a') as log:
            log.write(stage + chr(10) + result.stdout + chr(10))
        if result.returncode:
            print(result.stdout[-16000:])
            raise RuntimeError(stage + ' failed. The actual pip error is printed above; full log: ' + str(log_path))
        print(stage + ': OK', flush=True)
    run_pip(['install', '--dry-run', *install_options, *PACKAGES], 'Checking dependency compatibility')
    run_pip(['uninstall', '-y',
        'tensorflow', 'tensorflow-cpu', 'tf-nightly', 'tf-nightly-cpu', 'jax', 'jaxlib',
        'opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'],
        'Replacing overlapping runtime packages')
    run_pip(['install', *install_options, *PACKAGES], 'Installing compatible packages')
    # A separate interpreter verifies imports without loading changed libraries into this kernel.
    smoke = subprocess.run([sys.executable, '-c',
        'import numpy,pandas,sklearn,cv2,torch,torchvision,timm; '
        'from transformers import ViTModel; import tensorflow as tf; '
        'print("IMPORT CHECK PASSED", "TensorFlow", tf.__version__, "PyTorch", torch.__version__); '
        'assert tf.reduce_sum(tf.constant([1.,2.])).numpy() == 3.'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with log_path.open('a') as log:
        log.write(smoke.stdout)
    if smoke.returncode:
        print(smoke.stdout[-16000:])
        raise RuntimeError('Package import check failed; see the actual error above.')
    print(smoke.stdout[-2000:])
    if loaded_before:
        Path('/content/ai_shield_restart_required.json').write_text(json.dumps({'pid': os.getpid()}))
        print('INSTALLATION COMPLETE. Restart the Colab session once, then continue at section 2.')
    else:
        print('INSTALLATION COMPLETE. Continue at section 2.')
'''))

modules = {name: (ROOT / name).read_text(encoding='utf-8') for name in (
    'src/ai_shield_adaptation.py', 'src/ai_shield_adaptation_models.py',
    'src/ai_shield_metrics.py', 'src/ai_shield_cf_inference.py',
    'src/ai_shield_full_adaptation.py',
    'scripts/AI_Shield_Setup_Resume_v1.py',
    'scripts/AI_Shield_Semantic_Compatibility_Fix_v1.py',
    'scripts/AI_Shield_Standalone_Batch_Fix_v1.py')}
modules['src/__init__.py'] = ''
hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
cell('markdown', '## 2. Load the code\nAll required research code is embedded below; no GitHub push is needed.')
cell('code', 'EMBEDDED_SOURCES = ' + repr(modules) + '\nEMBEDDED_HASHES = ' + repr(hashes) + '\n' + textwrap.dedent('''
    import sys, hashlib, os, json, importlib, runpy
    from pathlib import Path
    restart_marker = Path('/content/ai_shield_restart_required.json')
    if restart_marker.exists() and json.loads(restart_marker.read_text()).get('pid') == os.getpid():
        raise RuntimeError('Setup succeeded. Restart the Colab session once, then rerun from section 2 to load the installed libraries.')
    CODE_ROOT = Path('/content/ai_shield_full_adaptation_code')
    for relative, source in EMBEDDED_SOURCES.items():
        assert hashlib.sha256(source.encode()).hexdigest() == EMBEDDED_HASHES[relative]
        path = CODE_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
    sys.path.insert(0, str(CODE_ROOT))
    # Refresh our Python modules when this cell is rerun in an existing session.
    # Existing frames/RUN/partition stay in the notebook namespace.
    for key in list(sys.modules):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]
    importlib.invalidate_caches()
    setup_tools = runpy.run_path(str(CODE_ROOT / 'scripts/AI_Shield_Setup_Resume_v1.py'))
    compatibility_tools = runpy.run_path(str(CODE_ROOT / 'scripts/AI_Shield_Semantic_Compatibility_Fix_v1.py'))
    _AI_SHIELD_INTEGRATED = True
    print('Integrated fixes v2 loaded. Continue with the next cell.')
'''))
cell('code', '''
    from google.colab import drive
    drive.mount('/content/drive')
    import gc, json, platform, time
    import numpy as np
    import pandas as pd
    import torch
    from IPython.display import display
    from src.ai_shield_adaptation import (
        seed_everything, check_audit_parity, train_branch, write_json,
        historical_references, checked_csv, align_predictions, benchmark, sha256_file)
    from src.ai_shield_adaptation_models import load_original_models
    from src.ai_shield_full_adaptation import *

    CONFIG = globals().get('CONFIG', FullConfig())
    # Normally leave this alone: current RUN or the saved setup is reused.
    RESUME_RUN_NAME = globals().get('RESUME_RUN_NAME', None)
    STAGE_TO_LOCAL = True  # Faster training: copy verified image bytes to temporary runtime storage.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE != 'cuda':
        raise RuntimeError('Choose Runtime > Change runtime type > GPU, then reconnect.')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('The original semantic audit used TF32 on A100. Select A100 or L4 to reproduce those checks; this GPU lacks TF32.')
    seed_everything(CONFIG.seed)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Per branch:', CONFIG.head_epochs, 'head epochs +', CONFIG.partial_epochs, 'limited-backbone epochs')
''')
cell('markdown', '''
    ## 3. Verify the saved data and reserve three training roles
    This cell now saves setup automatically and reuses the current or saved experiment.
    On recovery, surviving local images are reused after hashing; missing images are copied and
    verified in a single pass. No separate repair or resume script needs to be pasted into the notebook.
    If multiple older experiments exist with no saved selection, set RESUME_RUN_NAME in section 2.
    The complete dataset contains **5,124 images**: 2,990 train, 640 validation, 640 test,
    and 854 in the GPT Image 2 held-out-generator comparison. These are your existing splits.

    Generated images come from QwenImageBench; real images come from Open Images V7.
    The five development generators are FLUX.2 max, GLM-Image, Qwen-Image-2.0-pro,
    Seedream-5.0 and nano-banana-2.0. GPT Image 2 remains excluded from fitting.
    "Qwen data" here means the multi-generator benchmark, not only Qwen-generated images.

    Within the 2,990 training images, roughly **70% train the networks, 15% calibrate scores,
    and 15% fit fusion**. Connected prompts, duplicates and source groups stay together,
    so the printed counts can differ from those percentages. Validation chooses checkpoints
    and thresholds. Test images never fit a model, calibration or threshold.

    CF used all 2,990 training images for its own network. This matches the overall data pool,
    **not per-network exposure, training compute or earlier pretraining costs**.
    The tests have already been inspected in earlier work: they are historical comparisons,
    not a new blind confirmation set.
''')
cell('code', '''
    setup_tools['prepare_notebook_setup'](globals(), EMBEDDED_SOURCES, EMBEDDED_HASHES)
    display(partition.groupby(['role', 'label']).size().rename('images').reset_index())
    display(pd.concat(frames.values()).groupby(['split', 'generator', 'label']).size().rename('images').reset_index())
    print('Setup ready. Results folder:', RUN)
''')
cell('markdown', '''
    ## 4. Restore the original system and check it
    Before training, reproduce the main branches' saved features on six reference images,
    replay the original 17-feature fusion on both historical tests, and check face/no-face cases.
    The fusion replay uses the original cached features; it is not an assertion that every live
    auxiliary output is bit-identical to the old notebook.

    The VAE originally sampled a random latent vector even in evaluation mode. We preserve
    that sampled architecture but now seed **one CPU sample by the image's SHA-256** on every split.
    This makes new evaluation and standalone predictions agree on the same bytes.
    Both refitted variants share these exact frozen features. The original saved reference
    keeps its old draws and old thresholds; it also differs in calibration and fusion.
    The frozen-versus-fine-tuned comparison isolates the effect of network adaptation.

    The existing image preprocessing is preserved, including no EXIF transpose in the old branches.
    CF retains its own preprocessing. The face gate still decides when OpenForensics is applicable;
    OpenForensics receives the full resized image, as in your original audited implementation.
    Any parity failure stops the experiment so it can be investigated before training.
    The integrated compatibility check uses the fast image processor and scopes cuDNN TF32
    to the ViT backbone. The original tolerances are unchanged. Only a passing live GPU check
    can update an older saved implementation; backup files and a change record are retained.
''')
cell('code', '''
    compatibility_tools['apply_fix'](globals())
    setup_tools['save_setup_state'](globals())
    # Keep earlier reports when these small live checks are repeated after recovery.
    check_reports = RUN
    if any((RUN / name).exists() for name in ('main_branch_parity.json', 'original_full_fusion_replay.json', 'face_parity.json')):
        check_reports = RUN / 'resume_checks' / uuid.uuid4().hex[:12]
        check_reports.mkdir(parents=True, exist_ok=False)
    write_json(check_reports / 'main_branch_parity.json', parity)
    replay = replay_original(CONFIG, frames, check_reports)
    auxiliary_model = FrozenAuxiliary(CONFIG)
    face_parity = verify_face_parity(auxiliary_model, CONFIG, frames, check_reports)
    print('Reference checks passed:', parity['n'], 'main-branch images;', len(face_parity), 'face/no-face images.')
    display(pd.DataFrame(replay))
''')
cell('code', '''
    cache_started = time.perf_counter()
    auxiliary = cache_auxiliary(auxiliary_model, frames, CONFIG, RUN)
    auxiliary_seconds = time.perf_counter() - cache_started
    print('Frozen auxiliary features cached for all', sum(map(len, frames.values())), 'images.')
    print('Seconds:', round(auxiliary_seconds, 1))
''')
cell('markdown', '''
    ## 5. Control experiment: keep the networks frozen and refit the combination
    Refit the EfficientNet calibration on its reserved images, then the full 17-feature logistic
    fusion on a different reserved subset. This measures how much improvement comes from
    adapting the score combination alone. Each output's **max-F1 threshold** is selected on validation.
    F1 balances generated-image recall and precision; also inspect the real-image false-positive rate.
    Scores are detector outputs, not demonstrated probabilities that an image is AI-generated.
''')
cell('code', '''
    frozen_calibrator, frozen_fusion, frozen_thresholds, frozen_selection = fit_full_variant(
        'frozen_full', models, processor, partition, frames, auxiliary, CONFIG, RUN, DEVICE)
    display(pd.DataFrame([frozen_selection['validation_metrics']]))
''')
cell('markdown', '''
    ## 6. Fine-tune the two main branches
    Each starts from your saved model. First train the head for three epochs, then allow the
    final backbone block to learn for three epochs at a smaller learning rate. Earlier blocks
    and their BatchNorm statistics remain frozen. VAE and OpenForensics are never trained.
    Each branch's best weights are selected by validation AUROC, with the original weights
    also eligible. If adaptation does not help validation, the selected weights can remain original.
    Training time, trainable parameter counts, validation history and checkpoint hashes are saved.

    A disconnected run retains its saved files. This first version does not automatically resume
    optimizer state: use a new run after an interrupted training stage, rather than mixing results.
''')
cell('code', '''
    training_reports = []
    branch_train = partition[partition.role == 'branch_train'].copy()
    for branch, model in models.items():
        report = train_branch(model, branch, processor, branch_train,
            frames['validation'], CONFIG, RUN, DEVICE)
        training_reports.append(report)
        print('Selected checkpoint saved for:', branch)
    display(pd.DataFrame(training_reports))
''')
cell('code', '''
    adapted_calibrator, adapted_fusion, adapted_thresholds, adapted_selection = fit_full_variant(
        'adapted_full', models, processor, partition, frames, auxiliary, CONFIG, RUN, DEVICE)
    selections = {'frozen_full': frozen_selection, 'adapted_full': adapted_selection}
    display(pd.DataFrame([{'variant': name, **value['validation_metrics']}
                          for name, value in selections.items()]))
''')
cell('markdown', '''
    ## 7. Lock the choices, then evaluate
    The preferred new candidate is selected by validation F1, then validation AUROC.
    Files and thresholds are locked before producing test metrics. The test results can inform
    your report but cannot be used to pick a different checkpoint, threshold or candidate in this run.
''')
cell('code', '''
    # Read reporting metadata explicitly; it must not depend on which setup path ran.
    audit = json.loads((CONFIG.audit_root / 'saved_pipeline_evaluation_report.json').read_text(encoding='utf-8'))
    selection_lock = lock_full_selection(RUN, selections)
    verify_full_lock(RUN)
    verify_input_lock(CONFIG, RUN)
    print('Validation-selected candidate:', selection_lock['validation_selected_candidate'])
    comparison_rows, generator_rows = evaluate_full_variant('adapted_full', models, processor,
        frames, auxiliary, adapted_calibrator, adapted_fusion, adapted_thresholds, CONFIG, RUN, DEVICE)
    original_models, original_processor = load_original_models(CONFIG)
    rows, subgroups = evaluate_full_variant('frozen_full', original_models, original_processor,
        frames, auxiliary, frozen_calibrator, frozen_fusion, frozen_thresholds, CONFIG, RUN, DEVICE)
    comparison_rows += rows
    generator_rows += subgroups
    del original_models, original_processor
    gc.collect()
    torch.cuda.empty_cache()
    comparison_rows += historical_references(frames, CONFIG, audit, RUN)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RUN / 'comparison.csv', index=False)
    display(comparison[['model', 'split', 'auc', 'f1', 'accuracy', 'fpr', 'threshold']])
    display(pd.DataFrame(generator_rows))
''')
cell('markdown', '''
    ## 8. Check standalone predictions and measure speed against CF
    First check that single-image inference agrees with the batched adapted results on six fixed
    test images. This checks implementation only; it does not select or train the model.
    Then time 64 fixed validation images on this same runtime: three warmups, three timed passes,
    batch size one. The full pipeline includes live VAE, face checks, file reading and fusion.
    Timing excludes loading weights, the API and provenance. Process RAM is not isolated model memory.
    The semantic branch uses the saved evaluation batch size internally by repeating the one uploaded
    image and keeping one result. This avoids the observed singleton numerical discrepancy. Its extra
    computation is included in timing and explicitly recorded; one request still contains one image.
    The first cell includes a guarded repair for an already-trained session. It checks the six reference
    images and the last partial unseen-generator batch before saving any inference-code change.
''')
standalone_fix_source = (ROOT / 'scripts/AI_Shield_Standalone_Batch_Fix_v1.py').read_text(encoding='utf-8')
cell('code', '# Standalone batch compatibility fix: no retraining; original tolerances.\nexec(' + repr(standalone_fix_source) + ')\n' + textwrap.dedent('''
    predictor = FullPredictor(models, processor, auxiliary_model, adapted_calibrator, adapted_fusion, DEVICE,
        semantic_batch_size=CONFIG.batch_size)
    reference_images = frames['test'].sort_values('sample_id').groupby('label', group_keys=False).head(3)
    saved = checked_csv(RUN / 'adapted_full_test_predictions.csv').set_index('sample_id')
    standalone_checks = []
    for row in reference_images.itertuples():
        actual = predictor.predict(row.local_path)
        expected = float(saved.loc[row.sample_id, 'fusion_prob'])
        if not np.isclose(actual, expected, atol=5e-4, rtol=1e-4):
            raise ValueError('Standalone prediction differs: ' + row.sample_id)
        standalone_checks.append({'sample_id': row.sample_id, 'batch_score': expected,
                                  'standalone_score': actual, 'absolute_error': abs(actual - expected)})
    write_json(RUN / 'standalone_inference_parity.json', standalone_checks)
    display(pd.DataFrame(standalone_checks))

    timing_images = frames['validation'].sort_values('sample_id').groupby('label', group_keys=False).head(32)
    paths = timing_images.local_path.tolist()
    timing = {'device': DEVICE, 'gpu': torch.cuda.get_device_name(0), 'platform': platform.platform(),
              'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
              'sample_ids': timing_images.sample_id.tolist()}
    timing['adapted_full'] = benchmark(predictor.predict, paths, DEVICE)
    timing['adapted_full']['semantic_internal_batch_size'] = predictor.semantic_batch_size
    timing['adapted_full']['logical_request_batch_size'] = 1
    timing['adapted_full']['batch_note'] = 'The semantic branch repeats the input internally; all repeated computation is included in the measured latency and memory.'
    for model in models.values():
        model.cpu()
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
'''))
cell('code', '''
    from src.ai_shield_cf_inference import FrozenCF
    cf = FrozenCF(CONFIG.cf_root / 'checkpoints/stage2_last_blocks/best.pt',
        expected_sha256='51532d28870706a3c380a29889f634c556583bd62ed74775e5e610d5fd9d7b5f',
        threshold=0.6339887976646423,
        expected_manifest_sha256='2ba189a3be70e6840b6be6b530253925e87d7782c2882f0189ce8e582f20d01d',
        device=DEVICE)
    cf_live = pd.DataFrame([{'sample_id': row.sample_id, 'label': row.label,
        **cf.predict(row.local_path)} for row in timing_images.itertuples()])
    cf_saved = checked_csv(CONFIG.cf_root / 'predictions/finetuned_validation.csv')
    cf_saved = align_predictions(timing_images, cf_saved[cf_saved.sample_id.isin(timing_images.sample_id)], ['prob_fake'])
    if not np.allclose(cf_live.prob_fake, cf_saved.prob_fake, atol=1e-5, rtol=1e-5):
        raise ValueError('CF runtime parity failed')
    timing['fine_tuned_cf'] = benchmark(cf.predict, paths, DEVICE)
    write_json(RUN / 'runtime_comparison.json', timing)
    del cf
    gc.collect()
    torch.cuda.empty_cache()
    display(pd.DataFrame([{'model': name, **timing[name]} for name in ('adapted_full', 'fine_tuned_cf')]))
''')
cell('markdown', '''
    ## 9. Save the report and interpret it honestly
    - **Fine-tuned vs frozen/refitted:** did adapting the main networks add value?
    - **Full fusion vs each main branch:** did combining signals help or hurt?
    - **Full pipeline vs CF:** compare F1, AUROC, real-image false alarms, per-generator recall and latency.
    - A lower-cost or faster method can still be useful; small differences need repeat runs and uncertainty estimates.

    These splits have source/format shortcuts: generated images are square, and the GPT Image 2
    set is PNG while its real comparison is JPEG. Repeated prompts also mean images are not all
    independent examples. This notebook does not prove those shortcuts have been removed.
    Next checks are matched JPEG/resize conditions, human-created artwork/renders/edited photos,
    and a fresh generator/content set that has not guided development. Do not claim general image
    authenticity or replacement of CF from one good table.
''')
cell('code', '''
    verify_full_lock(RUN)
    verify_input_lock(CONFIG, RUN)
    report = {'status': 'completed', 'run': str(RUN), 'scope': 'full_17_feature_pipeline',
        'validation_selected_candidate': selection_lock['validation_selected_candidate'],
        'training': training_reports, 'comparison': comparison_rows, 'generator_breakdown': generator_rows,
        'main_branch_parity': parity, 'original_fusion_replay': replay, 'face_parity': face_parity,
        'standalone_parity': standalone_checks, 'auxiliary_cache_seconds': auxiliary_seconds,
        'runtime': timing, 'selection_lock': selection_lock,
        'caveats': ['Previously inspected historical tests; fresh confirmation remains necessary.',
                    'Same overall training pool as CF, not identical per-network exposure or compute.',
                    'New VAE draws are deterministic per image; the original reference retains its old draws.',
                    'Format/source/content shortcuts and non-photographic real images still need dedicated tests.',
                    'No demonstrated probability calibration or isolated peak RAM measurement.']}
    write_json(RUN / 'final_report.json', report)
    print('Done. Read comparison.csv and final_report.json in:')
    print(RUN)
    print('Selected research candidate:', report['validation_selected_candidate'])
''')
cell('markdown', '''
    ## Optional: try an uploaded image after the run
    This cell defaults to **off** so Run all completes without waiting for an upload.
    Turn it on and rerun this cell to try the validation-selected candidate.
    A score above its saved threshold is classified as generated; a score is not a certainty.
''')
cell('code', '''
    TRY_UPLOAD = False
    if TRY_UPLOAD:
        from google.colab import files
        variant = selection_lock['validation_selected_candidate']
        preview = load_full_predictor(CONFIG, RUN, DEVICE, variant=variant)
        threshold = selections[variant]['thresholds']['fusion_prob']
        for name, content in files.upload().items():
            path = Path('/content') / Path(name).name
            path.write_bytes(content)
            score = preview.predict(path)
            print(name, {'model': variant, 'generated_score': round(score, 4),
                         'threshold': threshold, 'label': 'generated' if score >= threshold else 'real'})
        del preview
        torch.cuda.empty_cache()
''')
cell('markdown', '''
    ## Reuse and sources
    Results live under `MyDrive/AI_Shield_Adaptation/full_pipeline_v1/<run_id>/`.
    Saved files include both fine-tuned checkpoints, each variant's calibration/fusion,
    thresholds, split roles, feature caches, per-image predictions, training histories and the final report.
    The `code/` folder contains the exact implementation. To reuse a **completed** run after installing
    dependencies and mounting Drive, add its `code` directory to `sys.path`, then call:
    ```python
    from src.ai_shield_full_adaptation import FullConfig, load_full_predictor
    predictor = load_full_predictor(FullConfig(), run_path, device='cuda', variant='adapted_full')
    score = predictor.predict(image_path)
    ```
    Use the variant and threshold recorded in `selection_lock.json` and its selection JSON.
    Original Drive artifacts remain required for loading the frozen auxiliary models.

    - [Original saved-pipeline audit](https://colab.research.google.com/drive/10jl7g0qknxLUCGF1VddbqbVf43oef4oD)
    - [Completed CF fine-tuning](https://colab.research.google.com/drive/1q5CB7k9fdY9ylfqMDQgClVl4k_ogNJef)
    - [Stacking and separate fitting data](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
    - [PyTorch evaluation and gradient modes](https://docs.pytorch.org/docs/stable/notes/autograd.html)
    - [Pinned TensorFlow release, including Python 3.13](https://pypi.org/project/tensorflow-cpu/2.20.0/)
    - [OpenCV package compatibility](https://pypi.org/project/opencv-python-headless/4.11.0.86/)
''')

notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'accelerator': 'GPU',
    'colab': {'name': NAME, 'provenance': []}}}
target = ROOT / 'notebooks' / NAME
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + '\n', encoding='utf-8')
print(target)
