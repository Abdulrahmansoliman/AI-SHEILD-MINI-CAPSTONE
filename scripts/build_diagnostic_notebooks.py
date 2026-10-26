"""Build three independent CPU Colabs; no inherited training-session globals."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'src/ai_shield_diagnostics.py').read_text(encoding='utf-8')
SOURCE_HASH = hashlib.sha256(SOURCE.encode()).hexdigest()


def build(number, name, title, description, sections):
    cells = []

    def cell(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        item = dict(cell_type=kind, id=f'diag-{number}-{len(cells):02d}', metadata={}, source=source.splitlines(True))
        if kind == 'code':
            compile(source, item['id'], 'exec')
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', f'''# {title}

{description}

**Run:** open this notebook in a fresh Colab tab, use a **CPU runtime** (no paid GPU),
then choose **Runtime → Run all** and approve the Drive mount. Each notebook is independent.
It reads the completed experiment `20261026T132144Z_58c9dad3`.

No PyTorch, TensorFlow, model downloads, image staging, or neural-network training.
Colab's standard NumPy, pandas, scikit-learn and Pillow packages are used without installation changes.
The original experiment, checkpoints, thresholds and app stay unchanged.

**Restart recovery:** run all cells again. These are small-table analyses; identical outputs are reused.
An interrupted write is retried. Changed inputs/code/library versions receive a new diagnostic folder.
Results are saved under `MyDrive/AI_Shield_Diagnostics/<training-run>/`.

**Limits:** the test sets have already been inspected. This is exploratory research, not a new blind
benchmark or a deployment decision. It does not test new image inference or JPEG/resize robustness.
''')
    cell('markdown', '''## 1. Mount Drive and load the completed run
This cell contains its own analysis code. It does not need the training notebook or any previous cell variables.
The checks verify saved tables and split identities, not every checkpoint or image byte.
''')
    cell('code', textwrap.dedent('''
        from google.colab import drive
        drive.mount('/content/drive')
        from pathlib import Path
        import hashlib, importlib.util, sys, tempfile
        from IPython.display import display
        import numpy as np
        import pandas as pd

        DRIVE_ROOT = Path('/content/drive/MyDrive')
        RUN_NAME = '20261026T132144Z_58c9dad3'
    ''') + '\nANALYSIS_SOURCE = ' + repr(SOURCE) + '\nEXPECTED_SOURCE_SHA256 = ' + repr(SOURCE_HASH) + '\n' + textwrap.dedent('''
        assert hashlib.sha256(ANALYSIS_SOURCE.encode()).hexdigest() == EXPECTED_SOURCE_SHA256
        code_dir = Path(tempfile.gettempdir()) / ('ai_shield_diagnostics_' + EXPECTED_SOURCE_SHA256[:12])
        code_dir.mkdir(parents=True, exist_ok=True)
        code_file = code_dir / 'analysis.py'
        code_file.write_text(ANALYSIS_SOURCE, encoding='utf-8')
        spec = importlib.util.spec_from_file_location('ai_shield_diagnostics_runtime', code_file)
        diag = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = diag
        spec.loader.exec_module(diag)
        data = diag.load_run(DRIVE_ROOT, RUN_NAME)
    ''') + f"\nOUT = diag.output_dir(data, {name!r})\n" + textwrap.dedent('''
        print('PASS: saved results reproduced; split identities and training roles verified.')
        print('Image models were not loaded. No training will be repeated.')
        display(data['roles'].groupby(['role', 'label']).size().rename('images').reset_index())
    '''))
    for kind, source in sections:
        cell(kind, source)
    cell('markdown', '''## Completion and next steps
All tables belong to this diagnostic folder; the trained run is read-only.
These findings can motivate a small, predefined experiment. Select models and thresholds using
training/validation only, then confirm once on a fresh test set. Avoid repeatedly tuning against the
historical test results printed here. Any image preprocessing or inference change needs new parity checks.
''')
    cell('code', '''
        diag.save_json(OUT / 'completed.json', {
            'status': 'completed', 'original_run': RUN_NAME,
            'analysis_version': diag.VERSION,
            'neural_network_training_performed': False,
            'deployment_changed': False,
            'scope': 'saved-feature diagnostics; historical tests are exploratory'
        })
        print('DONE. Results saved to:', OUT)
        print('You may disconnect this runtime. Run all again to regenerate the displays.')
    ''')
    notebook = dict(cells=cells, metadata={
        'colab': {'name': name + '.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'},
    }, nbformat=4, nbformat_minor=5)
    path = ROOT / 'notebooks' / (name + '.ipynb')
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(path)


def main():
    build('01', 'AI_Shield_01_Branch_Score_Audit_v1', 'AI Shield 01 — what is weak: features, training, or thresholds?',
        'Inspect the two branches, the saved fusion, and CF. Distinguish score-ranking quality from threshold/calibration problems.', [
        ('markdown', '''## 2. Reproduce the existing comparison
**Accuracy:** fraction correct. **F1:** balances precision and recall. **AUROC:** ranking quality across
thresholds (0.5 is chance-level ranking; 1 is perfect). **FPR:** fraction of real images incorrectly flagged.
**Brier:** squared score error, lower is better; it reflects both discrimination and calibration.
The validation numbers helped choose the model and are not independent test estimates.
'''),
        ('code', '''
            baseline = diag.baseline_table(data)
            diag.save_csv(OUT / 'baseline_metrics.csv', baseline)
            display(baseline[['split', 'model', 'accuracy', 'f1', 'auc', 'fpr', 'recall', 'brier', 'threshold']].round(4))
        '''),
        ('markdown', '''## 3. Did the networks stop improving?
Training loss and validation AUROC measure different things; do not subtract them.
Look for validation gains, plateaus, or deterioration while training loss falls.
Six epochs alone cannot prove either underfitting or overfitting. The saved checkpoint is the best validation
checkpoint, which may differ from the final epoch. Head and partial-backbone stages update different parameters.
'''),
        ('code', '''
            for branch, history in data['histories'].items():
                print(branch.upper())
                display(history.round(5))
                selected = next(r for r in data['report']['training'] if r['branch'] == branch)
                print('Selected:', selected['selected'])
                diag.save_csv(OUT / (branch + '_history.csv'), history)
        '''),
        ('markdown', '''## 4. Are scores separating the labels? Are they overconfident?
For each label, compare the median and 5th–95th percentile scores. Strong overlap suggests limited discrimination.
The fixed reliability bins compare mean model score with the observed generated fraction. For example, a bin
averaging 0.9 should contain roughly 90% generated images if scores are calibrated for this population.
Empty bins are omitted. Small bins are noisy; balanced benchmark calibration need not transfer to website traffic.
'''),
        ('code', '''
            distributions, reliability = diag.score_diagnostics(data)
            diag.save_csv(OUT / 'score_distributions.csv', distributions)
            diag.save_csv(OUT / 'reliability_bins.csv', reliability)
            display(distributions[distributions.split == 'test'].round(4))
            ece_rows = []
            for (split, model), group in reliability.groupby(['split', 'model']):
                ece = np.average(abs(group.mean_score - group.actual_generated_fraction), weights=group.n)
                ece_rows.append({'split': split, 'model': model, 'ECE_10_fixed_bins': float(ece)})
            ece_table = pd.DataFrame(ece_rows)
            diag.save_csv(OUT / 'calibration_summary.csv', ece_table)
            display(ece_table.round(4))
            display(reliability[(reliability.split == 'test') & (reliability.model == 'forensic')].round(4))
        '''),
        ('markdown', '''## 5. What happens if we prioritize fewer false accusations?
For each model, choose a threshold with at most **5% validation FPR**, maximizing validation recall.
Then apply that unchanged threshold to test sets. Test FPR may exceed 5%; this is not a guarantee.
This measures a trade-off, not a new model. It does not replace the app's saved threshold.
'''),
        ('code', '''
            threshold_choices, operating_points = diag.threshold_diagnostics(data)
            diag.save_json(OUT / 'validation_threshold_choices.json', threshold_choices)
            diag.save_csv(OUT / 'threshold_tradeoffs.csv', operating_points)
            display(operating_points[operating_points.split != 'validation'][
                ['split', 'model', 'operating_point', 'threshold', 'fpr', 'recall', 'accuracy', 'f1']].round(4))
        '''),
        ('markdown', '''## 6. Are many of the 17 features redundant?
Correlations are measured on the reserved fusion-training rows only. Highly correlated features may duplicate
information; this does not establish which should be removed. Notebook 02 tests that question directly.
'''),
        ('code', '''
            corr = data['train'][diag.FULL].corr(method='spearman')
            pairs = [{'feature_a': a, 'feature_b': b, 'spearman': float(corr.loc[a, b])}
                     for i, a in enumerate(diag.FULL) for b in diag.FULL[i+1:]
                     if np.isfinite(corr.loc[a, b])]
            correlations = pd.DataFrame(pairs).sort_values('spearman', key=abs, ascending=False)
            diag.save_csv(OUT / 'training_feature_correlations.csv', correlations)
            display(correlations.head(15).round(4))
            for message in diag.interpretation(data):
                print('-', message)
        '''),
    ])
    build('02', 'AI_Shield_02_Fusion_Ablations_v1', 'AI Shield 02 — which components actually help?',
        'Fit six small logistic classifiers on the saved fusion-training features. The image models remain frozen; no GPU is needed.', [
        ('markdown', '''## 2. Predefined comparisons
All six use the same **458 reserved fusion-training images**, StandardScaler and logistic regression
(`C=1`, seed 191). Scaling is fitted on training only. This is not a hyperparameter search.

| Candidate | Inputs |
|---|---|
| semantic_features | Semantic score plus its attention summaries |
| two_logits | One raw score from each main model |
| two_branches | Eight main-branch features; no VAE or face signals |
| full_without_vae | Full pipeline minus the VAE features |
| full_without_face | Full pipeline minus the face/OpenForensics features |
| full_refit | All 17 features: reproduction control |

The saved semantic-only, forensic-only, full pipeline and CF scores are also shown as references.
Removing a group requires refitting the combination; replacing it with zeros is not a valid ablation.
No CF-plus-semantic fusion is fitted: CF training scores would need an appropriate leakage-safe fitting protocol.
'''),
        ('code', '''
            display(pd.DataFrame([{'candidate': k, 'feature_count': len(v), 'features': ', '.join(v)}
                                  for k, v in diag.ABLATIONS.items()]))
            models, selection, validation_results = diag.fit_ablations(data)
            # Write and read back the choice BEFORE running historical test comparisons.
            diag.save_json(OUT / 'validation_selection.json', selection)
            diag.save_csv(OUT / 'validation_candidates.csv', validation_results)
            display(validation_results[['model', 'accuracy', 'f1', 'auc', 'fpr', 'threshold']].round(4))
            print('Full-refit control max error:', selection['full_refit_max_validation_error'])
            print('Validation-selected AI Shield variant:', selection['winner_among_AI_Shield_variants'])
            print('This is a research choice, not a deployment replacement.')
        '''),
        ('markdown', '''## 3. Apply the locked choices to historical tests
Each candidate keeps its validation-selected threshold. Do not pick a different winner after looking at this table.
These tests are exploratory because earlier work has already inspected them. A small difference does not
establish superiority; confirm promising changes on fresh data and inspect false positives.
'''),
        ('code', '''
            comparison, ablation_predictions = diag.evaluate_ablations(data, models, selection, OUT)
            display(comparison[comparison.split != 'validation'][
                ['split', 'model', 'accuracy', 'f1', 'auc', 'fpr', 'recall']].round(4))
            effects = []
            for split, group in comparison.groupby('split'):
                reference = group[group.model == 'full_refit'].iloc[0]
                for row in group.itertuples():
                    effects.append({'split': split, 'model': row.model,
                        'accuracy_change_pp_vs_full': 100 * (row.accuracy - reference.accuracy),
                        'f1_change_vs_full': row.f1 - reference.f1,
                        'fpr_change_pp_vs_full': 100 * (row.fpr - reference.fpr)})
            effects = pd.DataFrame(effects)
            diag.save_csv(OUT / 'component_effects.csv', effects)
            display(effects[effects.split == 'test'].round(4))
        '''),
        ('markdown', '''## 4. Inspect the learned combination
These are weights after standardizing the inputs. Positive weights push the generated score up.
They are not causal importance scores: correlated features can share or reverse weights.
Use the component-removal results above to judge whether a signal helps.
'''),
        ('code', '''
            weights = diag.coefficient_table(models)
            diag.save_csv(OUT / 'standardized_coefficients.csv', weights)
            display(weights[weights.model == 'full_refit'].sort_values(
                'standardized_weight', key=abs, ascending=False).round(4))
            # Export plain numerical parameters, not executable pickle files.
            parameters = {}
            for name, pipeline in models.items():
                parameters[name] = {'features': diag.ABLATIONS[name],
                    'scaler_mean': pipeline[0].mean_.tolist(), 'scaler_scale': pipeline[0].scale_.tolist(),
                    'coef': pipeline[-1].coef_[0].tolist(), 'intercept': float(pipeline[-1].intercept_[0]),
                    'classes': pipeline[-1].classes_.tolist(), 'threshold': selection['candidates'][name]['threshold']}
            diag.save_json(OUT / 'research_fusion_parameters.json', parameters)
            for message in diag.interpretation(data, comparison):
                print('-', message)
            print('Next: inspect disagreements in notebook 03; preserve a fresh confirmation set before any new tuning.')
        '''),
    ])
    build('03', 'AI_Shield_03_Failure_Review_v1', 'AI Shield 03 — inspect mistakes and complementary evidence',
        'Compare errors image by image, review generator/source groups, and display a small verified image gallery.', [
        ('markdown', '''## 2. Where do CF and the full adapted model disagree?
Use their original saved thresholds. `full_only_correct` means the adapted model is right and CF is wrong;
`CF_only_correct` means the reverse. The ground-truth label is used to describe errors, never to choose a
model for a live upload. Complementary errors do not demonstrate that an ensemble improves accuracy.
'''),
        ('code', '''
            errors = pd.concat([diag.error_table(data, split) for split in ('test', 'unseen_generator')], ignore_index=True)
            diag.save_csv(OUT / 'all_error_cases.csv', errors)
            summary = errors.groupby(['split', 'case']).size().rename('images').reset_index()
            diag.save_csv(OUT / 'disagreement_summary.csv', summary)
            display(summary)
            for split, frame in errors.groupby('split'):
                sem_corrects = int((frame.semantic_correct & ~frame.CF_correct).sum())
                cf_corrects = int((frame.CF_correct & ~frame.semantic_correct).sum())
                print(split, '| semantic correct / CF wrong:', sem_corrects,
                      '| CF correct / semantic wrong:', cf_corrects)
        '''),
        ('markdown', '''## 3. Which generators and file/source groups cause trouble?
For generated-only groups, report **recall**. For real-only groups, report **false-positive rate**.
Groups with fewer than 30 images are flagged. Filename extension is only a proxy for image encoding.
If label/source/format are strongly associated, the model could exploit shortcuts; this table alone cannot
prove that it does. No automatic artwork/edited-photo labels are invented when the manifest lacks them.
'''),
        ('code', '''
            subgroups = diag.subgroup_table(errors)
            diag.save_csv(OUT / 'subgroup_metrics.csv', subgroups)
            display(subgroups[(subgroups.field == 'generator') & (subgroups.model.isin(['saved_full', 'CF']))].round(4))
            formats = errors.groupby(['split', 'label', 'original_extension']).size().rename('images').reset_index()
            diag.save_csv(OUT / 'format_label_counts.csv', formats)
            display(formats)
            print('Available manifest fields:', ', '.join(data['manifests']['test'].columns))
        '''),
        ('markdown', '''## 4. Review a small image gallery
Shows up to three examples per error type and true label, separately for each test set.
Selection is deterministic by sample ID, so this is a case review, not a representative sample.
Only these images are opened from Drive. Their SHA-256 is checked before display.
Missing/unreadable/changed images are labelled and do not invalidate the tabular analysis.
The displayed scores are model outputs, not calibrated probabilities of authenticity.
'''),
        ('code', '''
            gallery_checks = []
            for split in ('test', 'unseen_generator'):
                selected = diag.select_gallery(errors[errors.split == split], per_case=3)
                diag.save_csv(OUT / (split + '_gallery_selection.csv'), selected)
                canvas, checks = diag.render_gallery(selected, DRIVE_ROOT)
                print(split)
                if canvas is not None:
                    display(canvas)
                checks['split'] = split
                gallery_checks.append(checks)
            # Availability can change after a Drive remount: use a content-addressed status file.
            checks = pd.concat(gallery_checks, ignore_index=True)
            check_hash = hashlib.sha256(checks.to_csv(index=False).encode()).hexdigest()[:12]
            diag.save_csv(OUT / ('gallery_image_status_' + check_hash + '.csv'), checks)
            display(checks.groupby(['split', 'status']).size().rename('images').reset_index())
        '''),
        ('markdown', '''## 5. Turn observations into a small experiment
The review sheet leaves image category and suspected cause blank for you to annotate. A visual impression is
a hypothesis, not proof of the mechanism. Check whether failures cluster around resizing, unusual lighting,
illustrations, edited real photographs, text, or a particular generator. Do not assume these categories
are covered simply because the benchmark contains many images.

Use notebook 01 to decide whether the issue is score ranking or the operating threshold; use notebook 02
to decide whether components help. Then change one thing, select on validation, and confirm on fresh data.
JPEG/resize inference tests and matched-budget neural-network retraining remain separate future experiments.
'''),
        ('code', '''
            review = errors[errors.case != 'both_correct'].copy()
            review['manual_image_category'] = ''
            review['suspected_failure_cause'] = ''
            review['evidence_or_notes'] = ''
            template = OUT / 'manual_review_template.csv'
            if not template.exists():
                diag.save_csv(template, review)
            else:
                print('Keeping your existing review sheet, including any manual annotations.')
            print('Manual review sheet:', template)
            for message in diag.interpretation(data):
                print('-', message)
        '''),
    ])


if __name__ == '__main__':
    main()
