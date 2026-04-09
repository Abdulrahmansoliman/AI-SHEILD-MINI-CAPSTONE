from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "code" / "PretrainedModel"
MODEL_STEM = MODEL_DIR / "dffnetv2B0"
MODEL_JSON = MODEL_STEM.with_suffix(".json")
MODEL_H5 = MODEL_STEM.with_suffix(".h5")
MODEL_ZIP = MODEL_STEM.with_suffix(".zip")
SAMPLE_DIR = MODEL_DIR / "streamlit_deepfake_detector" / "images"
RESULTS_CSV = BASE_DIR / "code" / "results" / "model_eval.csv"

TEST_METRICS = {
    "accuracy": 0.8577,
    "precision": 0.9188,
    "recall": 0.7828,
    "auc": 0.9387,
}


st.set_page_config(
    page_title="Faux Fighters Detector",
    page_icon="DF",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #f5f1e8;
            --paper: #fffdf8;
            --ink: #111111;
            --muted: #5b5b54;
            --accent: #cb4b16;
            --accent-2: #0b6e4f;
            --border: #d8d2c4;
            --shadow: rgba(17, 17, 17, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(203, 75, 22, 0.10), transparent 28%),
                radial-gradient(circle at bottom right, rgba(11, 110, 79, 0.10), transparent 26%),
                var(--bg);
            color: var(--ink);
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", sans-serif;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.02em;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,253,248,0.95), rgba(250,244,231,0.92));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.8rem 2rem;
            box-shadow: 0 18px 40px var(--shadow);
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 3rem;
            line-height: 1;
        }

        .hero p {
            color: var(--muted);
            margin-top: 0.8rem;
            margin-bottom: 0;
            font-size: 1.05rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .chip {
            display: inline-block;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.78);
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
            font-size: 0.9rem;
            color: var(--ink);
        }

        .panel {
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 10px 25px var(--shadow);
        }

        .scorecard {
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            height: 100%;
        }

        .scorecard h4 {
            margin: 0 0 0.4rem 0;
            font-size: 0.95rem;
            color: var(--muted);
            font-family: "IBM Plex Sans", sans-serif;
            font-weight: 600;
        }

        .scorecard .value {
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .eyebrow {
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            color: var(--accent);
            font-weight: 700;
        }

        .prob-wrap {
            margin-top: 1rem;
            display: grid;
            gap: 0.9rem;
        }

        .prob-label {
            display: flex;
            justify-content: space-between;
            font-weight: 600;
            margin-bottom: 0.25rem;
            font-size: 0.92rem;
        }

        .prob-track {
            height: 14px;
            border-radius: 999px;
            background: #ece7db;
            overflow: hidden;
        }

        .prob-fill-real {
            height: 100%;
            background: linear-gradient(90deg, #0b6e4f, #44b78b);
            border-radius: 999px;
        }

        .prob-fill-fake {
            height: 100%;
            background: linear-gradient(90deg, #8f250c, #cb4b16);
            border-radius: 999px;
        }

        .callout {
            margin-top: 1rem;
            border-left: 4px solid var(--accent);
            padding: 0.8rem 1rem;
            background: rgba(203, 75, 22, 0.06);
            border-radius: 0 14px 14px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_model_files() -> None:
    if MODEL_JSON.exists() and MODEL_H5.exists():
        return

    if not MODEL_ZIP.exists():
        raise FileNotFoundError(
            f"Missing model files and archive at {MODEL_ZIP}."
        )

    with zipfile.ZipFile(MODEL_ZIP, "r") as archive:
        archive.extractall(MODEL_DIR)


@st.cache_resource(show_spinner="Loading pretrained EfficientNetV2-B0...")
def load_model():
    try:
        from tensorflow.keras.models import model_from_json
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install the dependencies from requirements.txt before running inference."
        ) from exc

    ensure_model_files()
    with MODEL_JSON.open("r", encoding="utf-8") as handle:
        model = model_from_json(handle.read())
    model.load_weights(str(MODEL_H5))
    return model


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_CSV)
    if "models" not in frame.columns:
        frame = frame.rename(columns={frame.columns[1]: "models"})
    columns = ["models", "val_loss", "val_acc", "val_precision", "val_recall", "val_auc"]
    return frame[columns].sort_values("val_acc", ascending=False)


@st.cache_data(show_spinner=False)
def list_sample_images() -> dict[str, list[Path]]:
    samples: dict[str, list[Path]] = {}
    for label_dir in sorted(SAMPLE_DIR.iterdir()):
        if label_dir.is_dir():
            samples[label_dir.name] = sorted(
                path for path in label_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
    return samples


def preprocess_image(image_source) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(image_source)
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize((256, 256))
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return image, batch


def predict_image(model, image_source) -> dict[str, float | str | Image.Image]:
    display_image, batch = preprocess_image(image_source)
    real_probability = float(model.predict(batch, verbose=0)[0][0])
    real_probability = max(0.0, min(1.0, real_probability))
    fake_probability = 1.0 - real_probability
    label = "Real" if real_probability >= 0.5 else "Fake"
    confidence = max(real_probability, fake_probability)
    return {
        "image": display_image,
        "label": label,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": confidence,
    }


def render_probability_meter(real_probability: float, fake_probability: float) -> None:
    st.markdown(
        f"""
        <div class="prob-wrap">
            <div>
                <div class="prob-label"><span>Real probability</span><span>{real_probability:.1%}</span></div>
                <div class="prob-track"><div class="prob-fill-real" style="width:{real_probability * 100:.1f}%"></div></div>
            </div>
            <div>
                <div class="prob-label"><span>Fake probability</span><span>{fake_probability:.1%}</span></div>
                <div class="prob-track"><div class="prob-fill-fake" style="width:{fake_probability * 100:.1f}%"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(results: dict[str, float | str | Image.Image], caption: str) -> None:
    image_col, result_col = st.columns([1.15, 0.85], gap="large")

    with image_col:
        st.image(results["image"], caption=caption, use_container_width=True)

    with result_col:
        tone = "#0b6e4f" if results["label"] == "Real" else "#cb4b16"
        st.markdown(
            f"""
            <div class="panel">
                <div class="eyebrow">Prediction</div>
                <h2 style="margin-top:0.2rem; color:{tone};">{results['label']}</h2>
                <p style="color:#5b5b54; margin-top:-0.4rem;">Thresholded at 0.50 on the model's real-image probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_cols = st.columns(3)
        values = [
            ("Confidence", f"{results['confidence']:.1%}"),
            ("Real score", f"{results['real_probability']:.1%}"),
            ("Fake score", f"{results['fake_probability']:.1%}"),
        ]
        for column, (title, value) in zip(score_cols, values):
            with column:
                st.markdown(
                    f"""
                    <div class="scorecard">
                        <h4>{title}</h4>
                        <div class="value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        render_probability_meter(
            float(results["real_probability"]),
            float(results["fake_probability"]),
        )

        st.markdown(
            """
            <div class="callout">
                Best suited for photorealistic human imagery. The repository notes weaker generalization on harder manipulated test images.
            </div>
            """,
            unsafe_allow_html=True,
        )


def detector_tab(model) -> None:
    st.subheader("Upload an image")
    uploaded_image = st.file_uploader(
        "JPG, JPEG, PNG, or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        help="The original project focused on photorealistic human images.",
    )

    if uploaded_image is None:
        st.info("Upload an image to run inference with the pretrained EfficientNetV2-B0 model.")
        return

    results = predict_image(model, uploaded_image)
    render_prediction(results, "Uploaded image")


def sample_tab(model) -> None:
    samples = list_sample_images()
    label = st.radio(
        "Browse example images",
        options=list(samples.keys()),
        index=0,
        horizontal=True,
    )

    names = [path.name for path in samples[label]]
    selected_name = st.selectbox("Sample image", names)
    selected_path = next(path for path in samples[label] if path.name == selected_name)

    results = predict_image(model, selected_path)
    render_prediction(results, f"Sample: {selected_name}")
    st.caption(f"Ground truth folder label: {label}")


def about_tab() -> None:
    results = load_results().rename(
        columns={
            "models": "Model",
            "val_loss": "Val Loss",
            "val_acc": "Val Acc",
            "val_precision": "Val Precision",
            "val_recall": "Val Recall",
            "val_auc": "Val AUC",
        }
    )

    top = results.loc[results["Model"] == "effnetv2b0"].iloc[0]
    metric_cols = st.columns(4)
    metrics = [
        ("Validation accuracy", f"{top['Val Acc']:.4f}"),
        ("Validation precision", f"{top['Val Precision']:.4f}"),
        ("Validation AUC", f"{top['Val AUC']:.4f}"),
        ("Held-out test accuracy", f"{TEST_METRICS['accuracy']:.4f}"),
    ]
    for column, (title, value) in zip(metric_cols, metrics):
        column.metric(title, value)

    st.markdown("### Finalist comparison")
    formatted = results.copy()
    for column in ["Val Loss", "Val Acc", "Val Precision", "Val Recall", "Val AUC"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    st.markdown(
        """
        ### Deployment notes
        - This app loads a pretrained EfficientNetV2-B0 model from `code/PretrainedModel/dffnetv2B0.zip`.
        - If the extracted `.json` and `.h5` files are absent, the app unpacks them automatically on first run.
        - The original repository reports stronger validation performance than held-out test performance, so predictions should be treated as probabilistic, not definitive.
        """
    )


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Deepfake image classifier</div>
            <h1>Faux Fighters Detector</h1>
            <p>
                A deployable frontend for the repository's pretrained EfficientNetV2-B0 deepfake detector.
                Upload your own image, inspect sample predictions, and review the preserved experiment metrics.
            </p>
            <div class="chip-row">
                <span class="chip">EfficientNetV2-B0</span>
                <span class="chip">256 x 256 input</span>
                <span class="chip">Validation accuracy 0.9651</span>
                <span class="chip">Validation precision 0.9919</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model card")
        st.write("**Training data:** OpenForensics real/fake image dataset")
        st.write("**Input format:** 256 x 256 RGB")
        st.write("**Prediction rule:** score >= 0.50 => Real, otherwise Fake")
        st.write("**Best use case:** photorealistic human imagery")
        st.write("**Caution:** performance drops on harder manipulated test images")

    try:
        model = load_model()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    detector, samples, about = st.tabs(["Detector", "Sample gallery", "About"])
    with detector:
        detector_tab(model)
    with samples:
        sample_tab(model)
    with about:
        about_tab()


if __name__ == "__main__":
    main()
