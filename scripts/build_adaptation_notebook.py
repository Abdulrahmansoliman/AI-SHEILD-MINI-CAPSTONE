"""Build a self-contained Colab notebook from the tested research modules."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    entry = {"cell_type": kind, "id": f"adaptation-{len(cells):02d}", "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        compile(source, entry["id"], "exec")
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell("markdown", """
    # AI Shield: Forensic + Semantic Adaptation v1
    **Research only. The frozen CF detector, app and original checkpoints are never modified.**

    Question: how much can the two main networks improve after limited adaptation to Dataset v1?
    This is an **8-feature two-branch variant**, not a retraining of the original 17-feature pipeline.
    It uses EfficientNet-B0 and the original custom ViT attention head. VAE and face branches are excluded.
    The old full pipeline and original/fine-tuned CF are historical comparison rows, not fusion inputs.

    1. Check existing data/checkpoints and reproduce fixed audit predictions.
    2. Fit new fusion with both original networks frozen.
    3. Adapt each network for 3 head epochs + 3 limited-backbone epochs; refit fusion.
    4. Lock validation-selected checkpoints/thresholds, then evaluate the historical tests.
    5. Measure batch-1 latency against frozen CF on the same runtime and images.

    Of the original **2,990 training images**, about 80% train the branches and 20% train fusion.
    The exact split is by connected prompt/source/duplicate groups, so counts vary.
    Validation stays at 640. CF used all 2,990 for its own branch training: this comparison matches
    the available data budget, **not identical per-network exposure or compute**. Original pretraining costs are unknown.
    The already-inspected test and GPT Image 2 results are historical comparisons, not a fresh blind confirmation.

    Select a Colab GPU runtime, then run the cells in order. Mounting your Drive requires your approval.
    A real GPU run has not been performed just by generating this notebook. Failed checks must be investigated, not bypassed.
""")
requirements = (ROOT / "requirements-adaptation.txt").read_text().splitlines()
packages = [line for line in requirements if line.strip() and not line.startswith("#")]
cell("code", "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', " + ", ".join(repr(p) for p in packages) + "])")

modules = {name: (ROOT / name).read_text(encoding="utf-8") for name in (
    "src/ai_shield_adaptation.py", "src/ai_shield_adaptation_models.py",
    "src/ai_shield_metrics.py", "src/ai_shield_cf_inference.py",
)}
modules["src/__init__.py"] = ""
hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in modules.items()}
cell("markdown", "## 1. Load the embedded, versioned research code\nNo GitHub push, model upload or API changes are needed.")
cell("code", "EMBEDDED_SOURCES = " + repr(modules) + "\nEMBEDDED_HASHES = " + repr(hashes) + "\n" + textwrap.dedent("""
    import hashlib
    from pathlib import Path
    CODE_ROOT = Path('/content/ai_shield_adaptation_code')
    for relative, source in EMBEDDED_SOURCES.items():
        assert hashlib.sha256(source.encode()).hexdigest() == EMBEDDED_HASHES[relative]
        path = CODE_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
    sys.path.insert(0, str(CODE_ROOT))
    print('Research modules loaded. Original project files are untouched.')
"""))
cell("code", """
    from google.colab import drive
    drive.mount('/content/drive')
    import json, gc, platform
    import pandas as pd
    import torch
    from IPython.display import display
    from src.ai_shield_adaptation import *
    from src.ai_shield_adaptation_models import load_original_models

    CONFIG = Config()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(CONFIG.seed)
    print('Device:', DEVICE)
    print('GPU:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'No GPU attached')
    print('Training budget:', CONFIG.head_epochs, '+', CONFIG.partial_epochs, 'epochs per branch')
""")
cell("markdown", """
    ## 2. Verify inputs and reserve fusion-training images
    Reads the same Drive paths recorded by the completed CF and saved-baseline notebooks.
    Verifies split hashes, image bytes, labels, duplicates, group boundaries and original checkpoints.
    Creates a new timestamped folder under `MyDrive/AI_Shield_Adaptation/two_branch_v1/`.
    Run this cell only once for an experiment; starting it again creates a separate run.
""")
cell("code", """
    frames, audit = preflight(CONFIG)
    RUN, partition = prepare_run(CONFIG, frames)
    write_json(RUN / 'embedded_source_hashes.json', EMBEDDED_HASHES)
    display(partition.groupby(['role', 'label']).size().rename('images').reset_index())
    print('Output:', RUN)
""")
cell("markdown", """
    ## 3. Restore the original networks and check implementation parity
    Preserves the audited old-model preprocessing (no EXIF transpose), not CF's preprocessing.
    Loads the semantic head strictly and the pinned, originally frozen ViT backbone separately.
    Six fixed historical test images are used only for reproduction checking, never training or selection.
""")
cell("code", """
    models, processor = load_original_models(CONFIG)
    parity = check_audit_parity(models, processor, frames, CONFIG, DEVICE)
    write_json(RUN / 'baseline_parity.json', parity)
    print('Baseline parity PASS on', parity['n'], 'fixed images.')
""")
cell("markdown", """
    ## 4. Experiment A: frozen networks, newly fitted two-branch fusion
    Fits a standardized logistic regression on the reserved fusion-training images.
    Selects each score's max-F1 threshold on validation only. The old calibration is not reused.
    These are bounded detector scores, not established calibrated probabilities.
""")
cell("code", """
    frozen_fusion, frozen_thresholds = fit_variant(
        'frozen_pair', models, processor, partition, frames['validation'], CONFIG, RUN, DEVICE)
    print('Frozen-pair validation thresholds:', frozen_thresholds)
""")
cell("markdown", """
    ## 5. Experiment B: limited adaptation, then refit fusion
    Each branch starts from its original saved weights. Stage 1 trains the classifier/semantic head.
    Stage 2 also trains the final EfficientNet block + output convolution/normalization, or the final ViT block + normalization.
    Frozen BatchNorm statistics stay frozen. Checkpoints are chosen by validation AUROC, including the original as a candidate.
    Fusion images are excluded from branch training. No hyperparameter search is performed.
""")
cell("code", """
    if DEVICE != 'cuda':
        raise RuntimeError('Select a GPU runtime before full training. CPU unit tests are separate.')
    training_reports = []
    branch_train = partition[partition.role == 'branch_train'].copy()
    for branch, model in models.items():
        training_reports.append(train_branch(model, branch, processor, branch_train,
            frames['validation'], CONFIG, RUN, DEVICE))
    adapted_fusion, adapted_thresholds = fit_variant(
        'adapted_pair', models, processor, partition, frames['validation'], CONFIG, RUN, DEVICE)
    display(pd.DataFrame(training_reports))
""")
cell("markdown", """
    ## 6. Lock selections before final comparisons
    No choosing thresholds, epochs or fusion weights from test results.
    A lock prevents accidentally mixing checkpoints from different cell runs.
""")
cell("code", """
    selected_files = ['effnet_adapted.pt', 'semantic_adapted.pt', 'frozen_pair_fusion.joblib',
        'adapted_pair_fusion.joblib', 'frozen_pair_selection.json', 'adapted_pair_selection.json']
    selection_lock = {name: sha256_file(RUN / name) for name in selected_files}
    write_json(RUN / 'selection_lock.json', selection_lock)
    print('Selections locked. Historical evaluation can now run.')
""")
cell("code", """
    locked = json.loads((RUN / 'selection_lock.json').read_text())
    assert all(sha256_file(RUN / name) == value for name, value in locked.items())
    comparison_rows = evaluate_variant('adapted_pair', models, processor, frames,
        adapted_fusion, adapted_thresholds, CONFIG, RUN, DEVICE)
    original_models, original_processor = load_original_models(CONFIG)
    comparison_rows += evaluate_variant('frozen_pair', original_models, original_processor, frames,
        frozen_fusion, frozen_thresholds, CONFIG, RUN, DEVICE)
    del original_models, original_processor
    gc.collect()
    torch.cuda.empty_cache()
    comparison_rows += historical_references(frames, CONFIG, audit, RUN)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RUN / 'comparison.csv', index=False)
    display(comparison[['model', 'split', 'auc', 'f1', 'accuracy', 'fpr', 'threshold']])
""")
cell("markdown", """
    ## 7. Same-runtime timing: adapted pair versus frozen CF
    Uses 64 fixed validation images (32 per class), three warmups and three timed passes, batch size 1.
    Includes file reading, preprocessing, inference and score combination. Excludes loading weights, API and provenance work.
    Process RSS is not isolated peak model memory. These timings are not a public-deployment throughput guarantee.
""")
cell("code", """
    timing_images = frames['validation'].sort_values('sample_id').groupby('label', group_keys=False).head(32)
    paths = timing_images.local_path.tolist()
    timing = {'device': DEVICE, 'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
              'platform': platform.platform(), 'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
              'sample_ids': timing_images.sample_id.tolist()}
    pair = PairPredictor(models, processor, adapted_fusion, DEVICE)
    timing['adapted_pair'] = benchmark(pair.predict, paths, DEVICE)
    for model in models.values():
        model.cpu()
    del pair
    gc.collect()
    torch.cuda.empty_cache()

    from src.ai_shield_cf_inference import FrozenCF
    cf = FrozenCF(CONFIG.cf_root / 'checkpoints/stage2_last_blocks/best.pt',
        expected_sha256='51532d28870706a3c380a29889f634c556583bd62ed74775e5e610d5fd9d7b5f',
        threshold=0.6339887976646423,
        expected_manifest_sha256='2ba189a3be70e6840b6be6b530253925e87d7782c2882f0189ce8e582f20d01d',
        device=DEVICE)
    cf_live = pd.DataFrame([{'sample_id': row.sample_id, 'label': row.label,
        **cf.predict(row.local_path)} for row in timing_images.itertuples()])
    cf_saved = checked_csv(CONFIG.cf_root / 'predictions/finetuned_validation.csv')
    cf_saved = cf_saved[cf_saved.sample_id.isin(timing_images.sample_id)]
    cf_saved = align_predictions(timing_images, cf_saved, ['prob_fake'])
    assert np.allclose(cf_live.prob_fake, cf_saved.prob_fake, atol=1e-5, rtol=1e-5), 'CF runtime parity failed'
    timing['fine_tuned_cf'] = benchmark(cf.predict, paths, DEVICE)
    write_json(RUN / 'runtime_comparison.json', timing)
    del cf
    torch.cuda.empty_cache()
    display(pd.DataFrame([{'model': k, **timing[k]} for k in ('adapted_pair', 'fine_tuned_cf')]))
""")
cell("markdown", """
    ## 8. Final report and interpretation
    Compare frozen vs adapted networks, each branch vs the pair, and the pair vs CF.
    Check F1 AND real-image false alarms, not just AUROC. Small score differences need uncertainty analysis/repeat runs.
    Record a negative result honestly. A useful result may be cheaper adaptation without surpassing CF.
    No result automatically replaces the app detector. JPEG/resize robustness, non-photographic real images,
    matched real/generated content and a fresh confirmation set remain separate evaluation tasks.
""")
cell("code", """
    report = {
        'status': 'completed', 'run': str(RUN), 'scope': 'two_branch_variant_not_original_full_pipeline',
        'baseline_parity': parity, 'training': training_reports, 'comparison': comparison_rows,
        'runtime': timing, 'selection_lock': selection_lock,
        'caveats': ['Historical tests were previously inspected; fresh confirmation is still needed.',
            'Branches saw fewer images than CF because fusion training is reserved inside the 2990-image budget.',
            'No claim of matched pretraining cost, isolated memory peak or calibrated probabilities.',
            'Generated-vs-real source/content confounding is not resolved by this experiment.',
            'This is not a full 17-feature pipeline retraining; VAE and face branches are excluded.'],
    }
    write_json(RUN / 'final_report.json', report)
    print('Saved:', RUN / 'final_report.json')
    print('Frozen CF application remains unchanged.')
""")
cell("markdown", """
    ## Sources and implementation notes
    - Original model definitions: local `notebooks/phase2_semantic_report.ipynb` and the audited saved-baseline notebook.
    - [Saved baseline audit](https://colab.research.google.com/drive/10jl7g0qknxLUCGF1VddbqbVf43oef4oD)
    - [Completed CF experiment](https://colab.research.google.com/drive/1q5CB7k9fdY9ylfqMDQgClVl4k_ogNJef)
    - [Stacking and overfitting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
    - [Gradient modes and evaluation mode](https://docs.pytorch.org/docs/stable/notes/autograd.html)
""")
document = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU",
                         "colab": {"name": "AI_Shield_Forensic_Semantic_Adaptation_v1.ipynb", "provenance": []}}}
target = ROOT / "notebooks/AI_Shield_Forensic_Semantic_Adaptation_v1.ipynb"
target.write_text(json.dumps(document, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(target)
