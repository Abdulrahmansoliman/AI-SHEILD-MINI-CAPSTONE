# Forensic + Semantic Adaptation v1

## Current notebook: preserve the full saved pipeline

Use `notebooks/AI_Shield_Full_Pipeline_FineTuning_v1.ipynb` for the user's selected
experiment. It retains all 17 fusion features, including frozen VAE and face/OpenForensics
signals. It adapts EfficientNet and the custom semantic model, recalibrates EfficientNet,
and refits the full combination. The earlier eight-feature experiment below remains separate.

The full notebook reserves connected training groups for approximately 70% network
training, 15% calibration and 15% fusion. Validation selects checkpoints and thresholds;
the historical tests compare the original saved system, frozen/refitted system, adapted
system and CF. A frozen/refitted control separates network adaptation from fusion changes.
The VAE still samples a latent vector, but its sample is now deterministic per image hash
on every split and in standalone inference. Original historical reference scores retain
their original draws. The two newly refitted variants share identical auxiliary features.

Outputs go to `MyDrive/AI_Shield_Adaptation/full_pipeline_v1/<run_id>/`, including code,
checkpoints, calibration/fusion models, predictions, thresholds, histories and timings.
Use a fresh Colab Python 3.11–3.13 GPU runtime. TensorFlow's frozen face model runs on CPU.
Setup now uses TensorFlow 2.20, checks dependency resolution before uninstalling packages,
preserves Colab's torch/torchvision versions, and checks imports in a fresh process.
There is no automatic interrupted-training resume. Full real-data GPU execution remains
necessary; local verification is not a new accuracy result.

Build with `python scripts/build_full_adaptation_notebook.py`. The supporting module is
`src/ai_shield_full_adaptation.py`; checks are in `tests/test_full_adaptation.py` and
`tests/test_adaptation.py`. Local validation passed 31 tests, and the actual original
fusion checkpoint reproduced all 1,494 saved historical scores to floating-point precision.

## Earlier two-branch experiment

This experiment asks whether adapting the original two main networks improves their
new-data detection and how their runtime compares with fine-tuned CF. The application,
CF freeze, original checkpoints, and previous results are not modified.

## Run

Open `notebooks/AI_Shield_Forensic_Semantic_Adaptation_v1.ipynb` in Colab with a GPU.
Approve the Drive mount, then run cells in order. Code is embedded: no repository
upload or GitHub push is required. The notebook installs research dependencies only
in its Colab runtime. It records the actual torch/torchvision versions provided there.

Inputs are the existing `AI_Shield_Data/splits/v1` files, Phase 1 EfficientNet weights,
Phase 2 semantic weights, completed CF artifacts, and `saved_pipeline_v1/results` audit.
Paths come from the actual completed Colab notebooks. The preflight checks existence
and hashes; a mounted Colab execution is still needed to validate all private inputs.
Only the user's original, hash-verified checkpoints are loaded with legacy pickle support.

Outputs use a new timestamped directory under
`MyDrive/AI_Shield_Adaptation/two_branch_v1/`. Repeated training into an existing
checkpoint or selection lock fails. There is no automatic interrupted-run resume:
start a new run; keep incomplete results as incomplete. Never bypass a parity gate.

## What is compared

1. Original EfficientNet-B0 and original custom semantic attention model individually.
2. Their newly fitted two-branch fusion, with both networks unchanged.
3. Adapted EfficientNet and adapted semantic model individually.
4. Newly fitted fusion of the adapted models.
5. Original CF, frozen fine-tuned CF, and old full AI Shield as historical references.

**Scope correction:** this first notebook uses eight features from the two main
networks. It is not a retraining of the old 17-feature system. The VAE, face detector,
OpenForensics branch, and old Platt calibrator are not inputs to the new combination.
Comparing frozen vs adapted two-branch variants holds that architecture choice fixed.
The historical old-full row must not be described as the exact before-version of this variant.

## Data and selection

- Verify the recorded CF CSV hashes and all 5,124 image hashes; no silent file skipping.
- Reserve about 20% of the 2,990 training images for fusion; group prompts, sources,
  and exact duplicate bytes together. Train the networks only on the remainder.
- Use the original 640 validation images for epoch and max-F1 threshold selection.
- Use three head epochs and three limited-backbone epochs per branch; no automatic search.
- The original weights remain a candidate: worsening adaptation need not replace them.
- Fit `StandardScaler -> LogisticRegression` only on the reserved fusion subset.
- Lock checkpoints/fusion/thresholds before the 640-image test and 854-image held-out evaluation.
- Check six fixed historical examples for inference parity before training. This
  implementation check is explicitly separate from model selection.

The available data budget matches CF, but individual networks see fewer images, and
training time/compute is not matched. This is not an architecture-only causal comparison.
Previously inspected historical tests are not fresh blind tests. Original pretraining
costs are not included in newly measured adaptation time.

## Measurements and limits

Saves per-image scores, AUROC, F1, false-positive rate, accuracy, exact thresholds,
training histories, trainable/total parameter counts, new checkpoint hashes, and timings.
Timing uses the same 64 validation images, batch size one, three warmups and three passes.
CF's live scores must match its saved predictions before its latency is accepted.
Latency includes file decode and preprocessing, but excludes model load/API/provenance.
CUDA allocated memory is recorded; whole-notebook RSS is not isolated peak model RAM.

No real-data training or new accuracy claim is implied by unit-test success. JPEG/resize
robustness, genuine artwork/renders/edited images, source/content-confounding checks,
confidence intervals, seed variability, and a fresh confirmation set remain future work.

## Code and tests

- `src/ai_shield_adaptation.py`: data gates, split allocation, training, fusion and reporting.
- `src/ai_shield_adaptation_models.py`: verbatim original custom head and strict audited loaders.
- `scripts/build_adaptation_notebook.py`: deterministic notebook packaging, with source hashes.
- `tests/test_adaptation.py`: leakage, identity joins, thresholds, locks, notebook schema,
  frozen BatchNorm, gradient boundaries and a small synthetic training smoke test.

After editing sources, run `python scripts/build_adaptation_notebook.py`, then
`python -m unittest discover -s tests -p test_adaptation.py -v` in a compatible research
environment with PyTorch/torchvision, `requirements-adaptation.txt`, and `nbformat`.

Method references: [stacked-model leakage](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html),
[PyTorch gradient and evaluation modes](https://docs.pytorch.org/docs/stable/notes/autograd.html).
