# Deployment Guide

## Local Run

1. Put `ai_shield_deployment_bundle.zip` in the repository root.
2. Create and activate a Python 3.11 environment.
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the final AI Shield app from the repository root:

```powershell
streamlit run streamlit_app.py
```

The app extracts `ai_shield_deployment_bundle.zip` into `deployment_artifacts/` on first run. It then loads:

- Phase 1 EfficientNet-B0 forensic checkpoint
- Phase 1 VAE anomaly checkpoint
- OpenForensics Keras face model
- Phase 2 semantic ViT attention checkpoint
- Phase 3 full forensic + semantic GLM fusion model

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app and point it at:
   - repository: this repo
   - branch: your deployment branch
   - main file path: `streamlit_app.py`
3. In **Advanced settings**, choose **Python 3.11**.
4. Keep `requirements.txt` in the repo root so Streamlit installs the ML stack.
5. Make sure the deployment bundle is available. A 550MB zip is too large for normal GitHub commits, so use Git LFS, a private release asset, or run the app locally with the zip beside the code.

## Notes

- The model stack is heavy because it loads PyTorch, TensorFlow, a ViT backbone, and the final sklearn GLM.
- The first run can take time while dependencies load and the artifact bundle extracts.
- The final label is probabilistic guidance, not ground truth. The app exposes branch evidence so the user can see why the GLM moved toward real or fake.
- `app.py` is the older OpenForensics-only demo. `streamlit_app.py` is the final AI Shield deployment entrypoint.
- If you already created the app with the wrong Python version, delete it and redeploy with Python 3.11.
