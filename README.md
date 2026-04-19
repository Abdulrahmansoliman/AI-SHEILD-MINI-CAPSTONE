# From Deepfake Detector to AI Shield

<p align="center">
  <img src="repo/assets/project_header_image.jpg" alt="Project header" width="520">
</p>

<p align="center">
  An iterative Computational Sciences capstone project that started as a deployable image-level deepfake detector and is now expanding toward a broader AI Shield for images first and video later.
</p>

<p align="center">
  <img src="repo/assets/deepfake.gif" alt="App demo" width="760">
</p>

## Overview

This repository currently contains two connected tracks:

1. A completed, deployed image detector built on OpenForensics and EfficientNetV2-B0.
2. A new phase-1 pilot pipeline built on Tiny-GenImage in Google Colab, which is the first concrete step toward a broader multi-layer AI Shield.

The project changed after stress-testing revealed that high benchmark performance on face-centered real/fake classification is not the same as solving authenticity. Cases such as Gandhi taking a selfie, Photoshop-style edited portraits, and AI-edited photos of people showed that a stronger system needs more than one classifier.

That is why the longer-term roadmap now has four image phases:

- Phase 1: Low-level forensics
- Phase 2: Semantic consistency
- Phase 3: Context verification
- Phase 4: Provenance

The video branch comes after the image pipeline is stable.

---

## What Is In The Repo Right Now

| Track | Status | Data | Main model | Main artifact |
|---|---|---|---|---|
| OpenForensics detector | Completed and deployed | OpenForensics | EfficientNetV2-B0 | `code/PretrainedModel/dffnetv2B0.zip` |
| Tiny-GenImage pilot | Experimental, first new-pipeline run | Hugging Face `TheKernel01/Tiny-GenImage` | EfficientNet-B0 | `best.pt` |
| Streamlit app | Usable | Uses the OpenForensics detector | TensorFlow / Keras | `app.py` |
| Tiny-GenImage notebook | Reproducible training notebook | Google Colab + Google Drive | PyTorch / timm | `phase1_tiny_genimage_effnet_vae (3).ipynb` |

Important distinction:

- The deployed app currently uses the OpenForensics EfficientNetV2-B0 model.
- The Tiny-GenImage pilot weights in `best.pt` are not yet wired into the app.

---

## Results At A Glance

### 1. Completed OpenForensics detector

The strongest preserved model from the first iteration is EfficientNetV2-B0.

| Metric | Validation | Held-out test |
|---|---:|---:|
| Accuracy | 0.9651 | 0.8577 |
| Precision | 0.9919 | 0.9188 |
| Recall | 0.9498 | 0.7828 |
| AUC | 0.9824 | 0.9387 |

These results come from the saved evaluation artifacts in:

- `code/results/model_eval.csv`
- `code/results/v2b0_history.csv`

### 2. Tiny-GenImage pilot

The new image pipeline was first tested in Google Colab on a balanced pilot subset:

- 4,000 training images
- 1,000 validation images
- trained for 8 epochs

Pilot validation result:

- Validation AUC: 0.8503
- Validation accuracy: 0.7710

This is a pipeline-establishment run, not the final target. The full Tiny-GenImage scale discussed in the notebook is 28K+ images, and the current pilot was intentionally constrained by time and storage limits.

---

## Why The Project Changed

The first detector answered a narrower question:

> "Does this image look statistically similar to the real and fake face images the model saw during training?"

That is useful, but it is not the same as:

> "Is this image authentic?"

The difference matters. A model can score an image as visually plausible while still missing:

- semantic inconsistencies
- contextual impossibilities
- provenance problems
- non-face edit patterns outside its training distribution

This is why the project now uses an AI Shield framing rather than pretending one benchmark classifier solves the whole problem.

---

## Data Sources

### OpenForensics

Used for the first completed detector.

- Focus: real vs fake face-centered imagery
- Role in this repo: deployed first iteration and benchmark baseline
- Reference: `Le et al., ICCV 2021`

### Tiny-GenImage

Used for the new phase-1 pilot.

- Source: Hugging Face dataset `TheKernel01/Tiny-GenImage`
- Role in this repo: broader real-vs-AI image pilot for the redesigned pipeline
- Training environment: Google Colab with Google Drive artifact storage

Planned future data additions:

- edited-image benchmarks for Photoshop-style and AI-edited photos
- provenance-oriented data and metadata-rich sources
- later video datasets for temporal detection

---

## Run The Current App

The current Streamlit app loads the OpenForensics EfficientNetV2-B0 detector.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the app

```bash
streamlit run app.py
```

### 3. Model loading behavior

The app expects the deployed TensorFlow model in:

- `code/PretrainedModel/dffnetv2B0.zip`

If the extracted files are missing, the app automatically unpacks:

- `code/PretrainedModel/dffnetv2B0.json`
- `code/PretrainedModel/dffnetv2B0.h5`

Deployment notes are in:

- [`DEPLOYMENT.md`](DEPLOYMENT.md)

---

## Tiny-GenImage Pilot Notebook

The new-pipeline pilot was trained in:

- [`phase1_tiny_genimage_effnet_vae (3).ipynb`](phase1_tiny_genimage_effnet_vae%20(3).ipynb)

The notebook does the following:

- mounts Google Drive in Colab
- downloads Tiny-GenImage from Hugging Face
- creates a balanced subset for a practical first run
- trains an EfficientNet-B0 pilot
- saves:
  - `best.pt`
  - `last.pt`
  - `history.json`
  - `training_performance_v1.png`

Typical Colab packages for this notebook include:

- `torch`
- `torchvision`
- `timm`
- `datasets`
- `scikit-learn`
- `matplotlib`

The committed pilot weight file currently available in this repo is:

- `best.pt`

This notebook is the first implemented step of the redesigned AI Shield image pipeline.

---

## Repository Map

```text
deepfake-image-detector/
|-- app.py
|-- streamlit_app.py
|-- best.pt
|-- README.md
|-- DEPLOYMENT.md
|-- mini_capstone_authenticity_report.tex
|-- mini_capstone_part1 (1).tex
|-- phase1_tiny_genimage_effnet_vae (3).ipynb
|-- code/
|   |-- PretrainedModel/
|   |   |-- dffnetv2B0.zip
|   |   `-- streamlit_deepfake_detector/
|   |-- main/
|   |   |-- Training/
|   |   `-- Testing/
|   `-- results/
|-- repo/
|   |-- assets/
|   `-- screenshots/
`-- requirements.txt
```

---

## Reports And Capstone Documentation

This repo also contains the current written project materials:

- [`end_to_end_deepfake_report.tex`](end_to_end_deepfake_report.tex)
- [`mini_capstone_authenticity_report.tex`](mini_capstone_authenticity_report.tex)
- [`mini_capstone_part1 (1).tex`](mini_capstone_part1%20(1).tex)

These documents explain:

- the first OpenForensics detector
- the failure cases that changed the project
- the Tiny-GenImage pilot
- the four-phase AI Shield direction

---

## Limitations

Current limitations are important and intentional to state clearly:

- The deployed app is still tied to the OpenForensics model, which is face-centered.
- `best.pt` is committed, but it is not yet integrated into the app.
- The Tiny-GenImage run is a pilot subset, not a full-scale final training run.
- The broader AI Shield pipeline is partially implemented, not finished.
- The video branch is still a planned extension.

---

## Next Steps

Planned near-term work:

1. Move the Tiny-GenImage pilot artifacts into a cleaner model directory structure.
2. Add the next image-phase components:
   - variational autoencoder for anomaly-sensitive forensics
   - semantic consistency checks
   - context and provenance layers
3. Decide how the experimental `best.pt` model should be evaluated against the deployed OpenForensics baseline.
4. Extend the image-first system toward video once the image stack is stable.

---

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- PyTorch
- timm
- Hugging Face Datasets
- NumPy
- Pandas
- Pillow

---

## Notes For Contributors

If you are opening this repo fresh, the safest mental model is:

- Use the Streamlit app for the deployed first model
- Use the Tiny-GenImage notebook for the experimental second track
- Do not assume both models currently share the same inference pipeline

That separation is real and reflects the actual stage of the project.
