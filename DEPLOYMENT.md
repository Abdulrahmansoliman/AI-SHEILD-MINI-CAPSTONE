# Deployment Guide

## Local Run

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app from the repository root:

```powershell
streamlit run app.py
```

The app loads the pretrained model from `code/PretrainedModel/dffnetv2B0.zip`. If the extracted `.json` and `.h5` files are missing, it unpacks them automatically at startup.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app and point it at:
   - repository: this repo
   - branch: your deployment branch
   - main file path: `app.py`
3. In **Advanced settings**, choose **Python 3.11**.
4. Keep `requirements.txt` in the repo root so Streamlit installs TensorFlow, Streamlit, Pillow, NumPy, and Pandas.

## Notes

- The model is relatively heavy because it uses TensorFlow and a pretrained weight file.
- The original project reports higher validation performance than held-out test performance, so deployment should present predictions as probabilistic guidance rather than ground truth.
- The existing sample gallery under `code/PretrainedModel/streamlit_deepfake_detector/images` is already wired into the new app.
- If you already created the app with the wrong Python version, delete it and redeploy with Python 3.11.
