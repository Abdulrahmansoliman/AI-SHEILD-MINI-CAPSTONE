# Completed-pipeline diagnostics

Three independent CPU Colabs inspect run `20261026T132144Z_58c9dad3`:

Published and verified byte-for-byte against the tested local files:

- [01: Branch and score audit](https://colab.research.google.com/drive/1gGxbfwcCl6LmfAvpC8oMglg64jmP70zV)
- [02: Fusion ablations](https://colab.research.google.com/drive/1YQ1EZ2rdWc3PHXUY190FpEhKPJ4Zq0SC)
- [03: Failure review](https://colab.research.google.com/drive/1ZymZmplumtN4OsApAAc2bFhiBBnyzr7s)

1. `AI_Shield_01_Branch_Score_Audit_v1.ipynb`: reproduce results, inspect training histories, score overlap, fixed-bin calibration diagnostics, feature correlation, and validation-selected low-FPR thresholds.
2. `AI_Shield_02_Fusion_Ablations_v1.ipynb`: compare six predefined fusion feature sets on the existing 458 fusion-training examples. The image networks do not train again. Select thresholds and the research candidate on validation before evaluating historical tests.
3. `AI_Shield_03_Failure_Review_v1.ipynb`: compare per-image errors, generator/source/extension subgroups, and a small hash-verified image gallery. Export a review sheet without overwriting later annotations.

## Running and restarting

Open each notebook independently in Colab, choose CPU, and Run all. Approve the Drive mount. Standard Colab NumPy/pandas/scikit-learn/Pillow are used; no package installation or neural-model imports are performed. No helper file or variable from the training session is required.

The notebooks use the completed run, original split manifests, and saved fine-tuned CF predictions. They read approximately a few megabytes of tables, not checkpoints or all images. Notebook 03 opens up to 36 selected images only; missing or changed images receive placeholders and a status table.

Outputs go to `MyDrive/AI_Shield_Diagnostics/<run>/`. Each notebook uses a folder fingerprinted by input hashes, code and library versions. Rerunning identical work reuses identical files; changed evidence is not silently overwritten. A fresh runtime can Run all again. The original app, checkpoints, thresholds, and training experiment are unchanged.

## Verification and data isolation

- Split CSVs match the completed experiment's recorded hashes.
- IDs and labels align exactly; duplicates, partial joins, non-finite features and split/group/prompt overlaps are rejected.
- Fusion-training IDs must be the reserved `fusion_train` role. Calibration and network training stay separate.
- The saved full/CF test metrics and saved full validation metrics must reproduce.
- CSV floating-point values use round-trip parsing so a score exactly on the threshold stays on the threshold.
- Full-fusion refitting must reproduce saved validation scores before any ablation is accepted.
- Validation selection is saved before historical test evaluation. There is no threshold or hyperparameter search on test data.
- Source and library versions are recorded. No pickle/joblib model deserialization is required; research fusion parameters are exported as JSON.

Build: `python scripts/build_diagnostic_notebooks.py`.
Tests: `python -m unittest tests.test_diagnostics -v`.
Integration: `python scripts/verify_diagnostic_notebooks.py <local-MyDrive-mirror>`.

Local verification used the actual downloaded saved tables, all code cells in all three notebooks, twice each with fresh namespaces. Only the Drive mount/display were mocked and the root path changed. Eleven unit tests also cover threshold boundaries, isolation, reproducibility, notebook schema, and verified/missing/changed gallery images. This is not a claim of remote execution in the user's live Colab runtime.

## Initial findings from local execution

The full-refit validation control reproduced saved scores within `1.23e-15` maximum absolute error. The validation-selected AI Shield variant was `two_logits`, using only the forensic and semantic raw scores.

On historical tests, this simplified candidate had 82.66% main-test accuracy and 82.79% held-out-generator accuracy, versus 82.34% and 83.84% for the existing full combination. Its false-positive rates were lower (10.63% and 10.54%, versus 17.81% and 15.93%), but it missed more generated images. Its main-test F1 was 0.8141 versus 0.8237 for the full pipeline. This is a trade-off, not an across-the-board improvement. CF remained substantially stronger.

Changing fusion alone did not close the CF gap in these predefined comparisons. Results motivate targeted branch/data experiments rather than adding fusion complexity without evidence. Previously inspected tests are exploratory; any new deployment choice needs a fresh confirmation set.

These notebooks do not measure new-image inference, JPEG/resize robustness, new generators, EXIF/C2PA, or the proposed context-reasoning layer. They do not establish causal feature importance, unbiased confidence intervals, training-data equivalence with CF, or calibrated website probabilities.
