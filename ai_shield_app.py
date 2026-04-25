from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from ai_shield_inference import (
    AIShieldDependencyError,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BUNDLE_ZIP,
    FinalAIShieldInference,
    PredictionResult,
)


st.set_page_config(
    page_title="AI Shield Final Detector",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #121826;
            --muted: #536179;
            --line: #D8DEE9;
            --paper: #FFFFFF;
            --soft: #F6F8FB;
            --blue: #2563EB;
            --red: #DC2626;
            --green: #0F766E;
            --violet: #7C3AED;
        }
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        .app-title {
            font-size: 2.1rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: .25rem;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            max-width: 880px;
            margin-bottom: 1.1rem;
        }
        .status-strip {
            display: flex;
            gap: .55rem;
            flex-wrap: wrap;
            margin: .55rem 0 1.1rem 0;
        }
        .chip {
            border: 1px solid var(--line);
            background: var(--paper);
            color: var(--ink);
            border-radius: 999px;
            padding: .42rem .7rem;
            font-size: .82rem;
            font-weight: 650;
        }
        .result-panel {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFF 100%);
            border-radius: 10px;
            padding: 1.1rem 1.15rem;
            min-height: 100%;
        }
        .score-row {
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap:.75rem;
            margin:.9rem 0 .2rem 0;
        }
        .score-box {
            border: 1px solid var(--line);
            background: var(--paper);
            border-radius: 9px;
            padding: .85rem;
        }
        .score-label {
            color: var(--muted);
            font-size: .78rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .score-number {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.1;
            margin-top: .2rem;
        }
        .score-caption {
            color: var(--muted);
            font-size: .82rem;
            margin-top: .2rem;
        }
        .verdict {
            font-size: 2.5rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: 0;
            margin: .1rem 0 .25rem 0;
        }
        .verdict.fake { color: var(--red); }
        .verdict.real { color: var(--green); }
        .caption {
            color: var(--muted);
            font-size: .92rem;
        }
        .meter-label {
            display:flex;
            justify-content:space-between;
            color: var(--ink);
            font-size: .86rem;
            font-weight: 700;
            margin-top: .85rem;
            margin-bottom: .28rem;
        }
        .meter {
            height: 12px;
            border-radius: 999px;
            background: #E8EDF5;
            overflow: hidden;
            border: 1px solid #DCE3EE;
        }
        .meter-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--green), var(--blue), var(--red));
        }
        .branch-grid {
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: .75rem;
            margin-top: .35rem;
        }
        .branch-card {
            border: 1px solid var(--line);
            background: var(--paper);
            border-radius: 9px;
            padding: .9rem;
        }
        .branch-name {
            color: var(--muted);
            font-size: .78rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: .2rem;
        }
        .branch-score {
            color: var(--ink);
            font-size: 1.5rem;
            font-weight: 850;
            line-height: 1.1;
        }
        .branch-status {
            color: var(--muted);
            font-size: .82rem;
            margin-top: .18rem;
        }
        .section-note {
            border-left: 4px solid var(--blue);
            background: #F4F7FF;
            color: #26344D;
            padding: .85rem 1rem;
            border-radius: 8px;
            margin: .5rem 0 1rem 0;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_engine() -> FinalAIShieldInference:
    return FinalAIShieldInference.from_bundle(
        bundle_zip=DEFAULT_BUNDLE_ZIP,
        artifact_dir=DEFAULT_ARTIFACT_DIR,
    )


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def render_meter(label: str, value: float) -> None:
    value = max(0.0, min(1.0, float(value)))
    st.markdown(
        f"""
        <div class="meter-label"><span>{label}</span><span>{value:.1%}</span></div>
        <div class="meter"><div class="meter-fill" style="width:{value * 100:.1f}%"></div></div>
        """,
        unsafe_allow_html=True,
    )


def branch_card(name: str, score: float | None, status: str) -> str:
    return f"""
    <div class="branch-card">
        <div class="branch-name">{name}</div>
        <div class="branch-score">{pct(score)}</div>
        <div class="branch-status">{status}</div>
    </div>
    """


def final_metrics(metrics: dict) -> dict:
    return (
        metrics.get("models", {})
        .get("full_forensic_semantic_glm", {})
        .get("test", {})
    )


def metric_comparison(metrics: dict) -> pd.DataFrame:
    rows = []
    names = {
        "effnet_calibrated": "EfficientNet alone",
        "semantic_vit": "Semantic attention alone",
        "forensic_only_glm": "Phase 1 forensic GLM",
        "full_forensic_semantic_glm": "Final Phase 1 + Phase 2 GLM",
    }
    for key, label in names.items():
        test = metrics.get("models", {}).get(key, {}).get("test", {})
        if test:
            rows.append(
                {
                    "model": label,
                    "auc": test.get("auc"),
                    "accuracy": test.get("accuracy"),
                    "precision": test.get("precision"),
                    "recall": test.get("recall"),
                    "f1": test.get("f1"),
                    "threshold": test.get("threshold"),
                }
            )
    return pd.DataFrame(rows)


def render_result(result: PredictionResult) -> None:
    fake_class = "fake" if result.final_label == "Fake" else "real"
    meter_width = max(0.0, min(1.0, result.final_prob_fake)) * 100
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="caption">Final GLM decision</div>
            <div class="verdict {fake_class}">{result.final_label}</div>
            <div class="caption">
                Final fake probability is {result.final_prob_fake:.1%}.
                The validation-selected threshold is {result.threshold:.0%}.
            </div>
            <div class="score-row">
                <div class="score-box">
                    <div class="score-label">Phase 1 forensic-only GLM</div>
                    <div class="score-number">{result.forensic_prob_fake:.1%}</div>
                    <div class="score-caption">
                        Decision: {result.forensic_label} | threshold: {result.forensic_threshold:.0%}
                    </div>
                </div>
                <div class="score-box">
                    <div class="score-label">Final Phase 3 forensic + semantic GLM</div>
                    <div class="score-number">{result.final_prob_fake:.1%}</div>
                    <div class="score-caption">
                        Decision: {result.final_label} | threshold: {result.threshold:.0%}
                    </div>
                </div>
            </div>
            <div class="meter-label">
                <span>Final fake probability</span><span>{result.final_prob_fake:.1%}</span>
            </div>
            <div class="meter">
                <div class="meter-fill" style="width:{meter_width:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_branch_cards(result: PredictionResult) -> None:
    branches = result.branches
    html = '<div class="branch-grid">'
    html += branch_card("EfficientNet forensic", branches["effnet"].score, branches["effnet"].status)
    html += branch_card("VAE anomaly percentile", branches["vae"].score, branches["vae"].status)
    html += branch_card("Face gate", branches["face_gate"].score, branches["face_gate"].status)
    html += branch_card("OpenForensics face score", branches["openforensics"].score, branches["openforensics"].status)
    html += branch_card("Semantic ViT attention", branches["semantic"].score, branches["semantic"].status)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_feature_table(result: PredictionResult) -> None:
    feature_long = (
        result.features.T.reset_index()
        .rename(columns={"index": "feature", 0: "value"})
        .assign(value=lambda frame: frame["value"].astype(float))
    )
    st.dataframe(
        feature_long,
        use_container_width=True,
        hide_index=True,
        column_config={
            "feature": st.column_config.TextColumn("Feature"),
            "value": st.column_config.NumberColumn("Value", format="%.6f"),
        },
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-title">AI Shield Final Detector</div>
        <div class="app-subtitle">
        Upload one image and the deployed system runs the frozen Phase 1 forensic branches,
        the Phase 2 semantic attention branch, and the Phase 3 GLM meta-layer that combines their evidence.
        </div>
        <div class="status-strip">
            <span class="chip">Phase 1 forensic evidence</span>
            <span class="chip">Face-gated OpenForensics</span>
            <span class="chip">Phase 2 semantic attention</span>
            <span class="chip">Phase 3 GLM late fusion</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(engine: FinalAIShieldInference) -> None:
    st.sidebar.title("Model Stack")
    st.sidebar.caption("Artifacts are loaded from the deployment bundle.")
    st.sidebar.markdown(
        f"""
        **Artifact folder**  
        `{Path(engine.paths.artifact_dir).name}`

        **Decision threshold**  
        `{engine.threshold:.2f}`
        """
    )

    metrics = final_metrics(engine.load_metrics())
    if metrics:
        st.sidebar.divider()
        st.sidebar.caption("Final GLM test metrics")
        st.sidebar.metric("ROC-AUC", f"{metrics.get('auc', 0):.3f}")
        st.sidebar.metric("F1", f"{metrics.get('f1', 0):.3f}")
        st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")


def render_model_card(engine: FinalAIShieldInference) -> None:
    metrics = engine.load_metrics()
    st.markdown(
        """
        <div class="section-note">
        This deployed model is late fusion. It does not retrain the image models during inference.
        Each branch produces a small set of evidence features, then the GLM learns how to weight those features.
        </div>
        """,
        unsafe_allow_html=True,
    )
    table = metric_comparison(metrics)
    if not table.empty:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
            },
        )

    coefficients = engine.load_coefficients()
    if not coefficients.empty:
        st.subheader("GLM coefficient evidence")
        st.caption(
            "Positive coefficients push the final prediction toward fake after the feature scaling in the saved pipeline."
        )
        st.dataframe(coefficients, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    try:
        engine = load_engine()
    except Exception as exc:
        render_header()
        st.error(str(exc))
        st.info(
            "Place ai_shield_deployment_bundle.zip in the repository root, then restart the app."
        )
        return

    render_sidebar(engine)
    render_header()

    tab_run, tab_evidence, tab_model = st.tabs(["Run detector", "Evidence table", "Model card"])
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with tab_run:
        left, right = st.columns([0.92, 1.08], gap="large")
        with left:
            uploaded = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
                help="The app runs the frozen branches and the final GLM fusion model on this image.",
            )
            if uploaded is not None:
                image = Image.open(uploaded)
                st.image(image, caption=f"{uploaded.name}", use_column_width=True)
            else:
                st.info("Upload a JPG, PNG, or WEBP image to run the final detector.")

        with right:
            if uploaded is not None:
                try:
                    with st.spinner("Running forensic, semantic, and GLM fusion inference..."):
                        result = engine.predict(Image.open(uploaded))
                    st.session_state.last_result = result
                    render_result(result)
                    st.subheader("Branch evidence")
                    render_branch_cards(result)
                except AIShieldDependencyError as exc:
                    st.error(str(exc))
                    st.info("Install the full deployment dependencies from requirements.txt and restart.")
                except Exception as exc:
                    st.error("Inference failed before the final GLM could produce a score.")
                    st.exception(exc)
            elif st.session_state.last_result is not None:
                render_result(st.session_state.last_result)
                st.subheader("Branch evidence")
                render_branch_cards(st.session_state.last_result)

    with tab_evidence:
        result = st.session_state.last_result
        if result is None:
            st.info("Run the detector once to see the exact feature row sent into the GLM.")
        else:
            st.markdown(
                """
                <div class="section-note">
                This is the one-row feature table created for the uploaded image.
                The final GLM reads these values in the same feature order used during Phase 3.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_feature_table(result)

    with tab_model:
        render_model_card(engine)


if __name__ == "__main__":
    main()
