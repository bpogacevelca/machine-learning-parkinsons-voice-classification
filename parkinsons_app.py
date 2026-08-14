import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
from scipy.stats import entropy


# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="centered"
)


# ===================================================
# CUSTOM STYLING
# ===================================================

st.markdown(
    """
    <style>

    /* Overall page */
    .stApp {
        background-color: #DCE8FF;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Streamlit top header */
    [data-testid="stHeader"] {
        background-color: #AFCBF7 !important;
    }

    [data-testid="stToolbar"] {
        background-color: #AFCBF7 !important;
    }

    [data-testid="stDecoration"] {
        background-color: #AFCBF7 !important;
    }

    /* Main title */
    h1 {
        color: #111827 !important;
        text-align: center;
        font-weight: 800;
    }

    h2, h3 {
        color: #111827 !important;
        font-weight: 700;
    }

    p, li {
        color: #26364A;
        line-height: 1.6;
    }

    .subtitle {
        text-align: center;
        color: #26364A;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 28px;
    }

    /* Model performance cards */
    .metric-card {
        background-color: white;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        border: 1px solid #C7D7F2;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    .metric-card .big {
        font-size: 22px;
        font-weight: 800;
        color: #1E3A5F;
    }

    .metric-card .small {
        color: #4B5563;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Space under performance cards */
    .performance-gap {
        height: 22px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white !important;
        border-radius: 16px;
        padding: 15px;
    }

    [data-testid="stFileUploader"] section {
        background-color: white !important;
        border: 1px solid #BBD0F2 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #111827 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #4B5563 !important;
    }

    /* Brighter upload button */
    [data-testid="stFileUploader"] button {
        background-color: #4A90E2 !important;
        color: white !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.1rem !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #2F7DD1 !important;
        color: white !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: white !important;
        border-radius: 14px !important;
        border: none !important;
    }

    [data-testid="stExpander"] * {
        color: #111827 !important;
    }

    /* Analyse button */
    .stButton > button {
        background-color: #4A90E2;
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 0.65rem 1.3rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #2F7DD1;
        color: white !important;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ===================================================
# PATHS
# ===================================================

MODEL_PATH = "models/random_forest_model.joblib"
DATASET_PATH = "all_dataset.csv"

FIGSHARE_URL = (
    "https://figshare.com/articles/dataset/"
    "Voice_Samples_for_Patients_with_Parkinson_s_Disease_"
    "and_Healthy_Controls/23849127"
)


# ===================================================
# LOAD MODEL
# ===================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Could not load Random Forest model: {e}")
    st.stop()


# ===================================================
# LOAD TRAINING FEATURE ORDER
# ===================================================

try:
    training_df = pd.read_csv(DATASET_PATH)

    feature_columns = [
        column
        for column in training_df.columns
        if column not in ["status", "File_Name"]
    ]

except Exception as e:
    st.error(f"Could not load training dataset: {e}")
    st.stop()


# ===================================================
# FEATURE EXTRACTION
# ===================================================

def extract_features(file_path):

    try:
        y, sr = librosa.load(
            file_path,
            sr=None
        )

        features = {}

        # Fundamental frequency
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr
        )

        pitches = pitches[pitches > 0]

        features["MDVP:Fo(Hz)"] = (
            np.mean(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features["MDVP:Fhi(Hz)"] = (
            np.max(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features["MDVP:Flo(Hz)"] = (
            np.min(pitches)
            if len(pitches) > 0
            else np.nan
        )


        # Jitter features
        zero_crossings = librosa.zero_crossings(
            y,
            pad=False
        )

        jitter_std = np.std(zero_crossings)
        jitter_mean = np.mean(zero_crossings)

        features["MDVP:Jitter(%)"] = (
            jitter_std / jitter_mean
            if jitter_mean != 0
            else np.nan
        )

        features["MDVP:Jitter(Abs)"] = (
            jitter_std
            if jitter_std > 0
            else np.nan
        )

        features["MDVP:RAP"] = (
            jitter_std /
            (len(zero_crossings) + 1e-6)
        )

        features["MDVP:PPQ"] = (
            jitter_std /
            np.sqrt(len(zero_crossings) + 1e-6)
        )

        features["Jitter:DDP"] = (
            jitter_std * 3
        )


        # Shimmer features
        amplitude = librosa.amplitude_to_db(
            np.abs(y),
            ref=np.max
        )

        shimmer_std = np.std(amplitude)
        shimmer_mean = np.mean(amplitude)

        features["MDVP:Shimmer"] = (
            shimmer_std / shimmer_mean
            if shimmer_mean != 0
            else np.nan
        )

        features["MDVP:Shimmer(dB)"] = shimmer_std
        features["Shimmer:APQ3"] = shimmer_std / 3
        features["Shimmer:APQ5"] = shimmer_std / 5
        features["MDVP:APQ"] = shimmer_std / len(amplitude)
        features["Shimmer:DDA"] = shimmer_std * 3


        # Harmonic / noise features
        harmonic, percussive = (
            librosa.effects.hpss(y)
        )

        features["NHR"] = (
            np.mean(percussive) /
            (np.mean(harmonic) + 1e-6)
        )

        features["HNR"] = (
            np.mean(harmonic) /
            (np.mean(percussive) + 1e-6)
        )


        # Nonlinear features
        features["RPDE"] = (
            entropy(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features["DFA"] = (
            librosa.feature.rms(y=y).mean()
        )


        # Spread / PPE features
        features["spread1"] = (
            np.std(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features["spread2"] = (
            np.var(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features["D2"] = (
            np.percentile(pitches, 99)
            if len(pitches) > 0
            else np.nan
        )

        features["PPE"] = (
            np.mean(
                np.abs(
                    pitches -
                    np.mean(pitches)
                )
            )
            if len(pitches) > 0
            else np.nan
        )

        return features

    except Exception as e:

        st.error(
            f"Feature extraction failed: {e}"
        )

        return None


# ===================================================
# HEADER
# ===================================================

st.title(
    "🧠 Parkinson's Disease Detection"
)

st.markdown(
    """
    <div class="subtitle">
        Using Voice Recordings and Machine Learning
        for Parkinson's Disease Classification
    </div>
    """,
    unsafe_allow_html=True
)


# ===================================================
# ABOUT
# ===================================================

with st.container(border=True):

    st.subheader(
        "About this prototype"
    )

    st.write(
        """
        This educational machine-learning prototype analyses
        acoustic features from sustained vowel recordings and
        uses a Random Forest classifier to compare them with
        patterns learned from Parkinson's disease and
        healthy-control voice samples.
        """
    )

    st.write(
        """
        The application demonstrates an end-to-end workflow
        from audio feature extraction to machine-learning
        classification.
        """
    )


# ===================================================
# DISCLAIMER
# ===================================================

st.warning(
    """
    **Important: this is a student research prototype,
    not a diagnostic test.**

    The model was trained on a small research dataset under
    specific recording conditions. Different phones,
    microphones, rooms and audio formats may produce
    inaccurate classifications.

    The output should not be interpreted as evidence that
    someone does or does not have Parkinson's disease.
    """
)


# ===================================================
# RECORDING INSTRUCTIONS
# ===================================================

st.info(
    """
    🎙️ **Recording instructions**

    Record one steady, sustained pronunciation of **/a/**
    ("aaaaah") for approximately **3–5 seconds**
    in a quiet environment.
    """
)


# ===================================================
# UPLOAD
# ===================================================

st.subheader(
    "🎙️ Upload Voice Recording"
)


uploaded_file = st.file_uploader(
    "Upload audio",
    type=["wav", "m4a", "mp3"],
    label_visibility="collapsed"
)


# ===================================================
# ANALYSIS
# ===================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_audio:

        temp_audio.write(
            uploaded_file.getbuffer()
        )

        audio_path = temp_audio.name


    if st.button(
        "🔬 Analyse Voice",
        type="primary"
    ):

        with st.spinner(
            "Extracting acoustic features and running the model..."
        ):

            features = extract_features(
                audio_path
            )


        if features is not None:

            missing_features = [
                feature
                for feature in feature_columns
                if feature not in features
            ]


            if missing_features:

                st.error(
                    "The extracted features do not match "
                    "the features used to train the model."
                )


            else:

                feature_df = pd.DataFrame(
                    [features]
                )

                feature_df = feature_df[
                    feature_columns
                ]

                feature_df = feature_df.replace(
                    [np.inf, -np.inf],
                    np.nan
                )


                if feature_df.isnull().any().any():

                    st.error(
                        """
                        Some acoustic features could not be extracted
                        reliably from this recording.

                        Please try another clear recording.
                        """
                    )


                else:

                    try:

                        prediction = model.predict(
                            feature_df
                        )[0]

                        st.markdown("---")

                        st.subheader(
                            "📊 Model Classification"
                        )


                        if prediction == 1:

                            st.error(
                                "🔴 Parkinson's-class pattern"
                            )

                            st.caption(
                                """
                                The model placed this recording into
                                the Parkinson's-labelled category
                                learned from the training data.

                                This does **not** mean that the speaker
                                has Parkinson's disease.
                                """
                            )


                        else:

                            st.success(
                                "🟢 Healthy-control pattern"
                            )

                            st.caption(
                                """
                                The model placed this recording into
                                the healthy-control category learned
                                from the training data.

                                This is not a medical assessment.
                                """
                            )


                        with st.expander(
                            "🔍 View extracted acoustic features"
                        ):

                            display_df = (
                                feature_df
                                .T
                                .reset_index()
                            )

                            display_df.columns = [
                                "Feature",
                                "Value"
                            ]

                            st.dataframe(
                                display_df,
                                width="stretch",
                                hide_index=True
                            )


                    except Exception as e:

                        st.error(
                            f"Prediction failed: {e}"
                        )


    try:
        os.remove(audio_path)

    except Exception:
        pass


# ===================================================
# MODEL PERFORMANCE
# ===================================================

st.markdown("---")

st.subheader(
    "📈 Model Performance"
)

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
        <div class="metric-card">
            <div class="big">
                Random Forest
            </div>
            <div class="small">
                Final selected model
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="metric-card">
            <div class="big">
                73.97% ± 5.04%
            </div>
            <div class="small">
                Mean 5-fold CV accuracy
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="performance-gap"></div>',
    unsafe_allow_html=True
)


# ===================================================
# DATASET & SOURCE
# ===================================================

with st.expander(
    "📚 Dataset & source"
):

    st.write(
        """
        The voice recordings used to create the training
        dataset were obtained from a publicly available
        research dataset hosted on **Figshare**.

        The dataset contains voice samples from Parkinson's
        disease and healthy-control participants that were
        used for acoustic feature extraction and
        machine-learning classification in this project.
        """
    )

    st.link_button(
        "🔗 View original dataset on Figshare",
        FIGSHARE_URL
    )


# ===================================================
# LIMITATIONS
# ===================================================

with st.expander(
    "⚠️ Known limitations"
):

    st.markdown(
        """
        - The training dataset is small.
        - Cross-validation was performed within the available dataset.
        - The model has not been validated on an independent clinical cohort.
        - Smartphone microphones and recording environments may alter the extracted features.
        - Preliminary informal smartphone testing showed limited generalisation to new recordings.
        - The acoustic features used here should not be treated as validated clinical biomarkers.
        """
    )


# ===================================================
# GENERALISATION
# ===================================================

with st.expander(
    "💡 Why can new recordings be classified incorrectly?"
):

    st.write(
        """
        Machine-learning models learn patterns from the
        data used during training.

        If a new recording differs because of microphone
        type, audio compression, room acoustics, recording
        distance or speaker characteristics, the extracted
        features may also differ.

        This is a **generalisation problem**.

        A stronger future version would use a larger,
        more diverse dataset, standardised recording
        procedures and independent external validation.
        """
    )


# ===================================================
# FOOTER
# ===================================================

st.markdown("---")

st.caption(
    """
    Research and educational use only.
    This application is not a medical device,
    screening test or diagnostic system.
    """
)