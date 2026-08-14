import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy


# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = {}

        # Fundamental frequencies
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]

        features['MDVP:Fo(Hz)'] = np.mean(pitches) if len(pitches) > 0 else np.nan
        features['MDVP:Fhi(Hz)'] = np.max(pitches) if len(pitches) > 0 else np.nan
        features['MDVP:Flo(Hz)'] = np.min(pitches) if len(pitches) > 0 else np.nan

        # Jitter-related features
        zero_crossings = librosa.zero_crossings(y, pad=False)

        jitter_std = np.std(zero_crossings)
        jitter_mean = np.mean(zero_crossings)

        features['MDVP:Jitter(%)'] = (
            jitter_std / jitter_mean
            if jitter_mean != 0
            else np.nan
        )

        features['MDVP:Jitter(Abs)'] = (
            jitter_std
            if jitter_std > 0
            else np.nan
        )

        features['MDVP:RAP'] = jitter_std / (len(zero_crossings) + 1e-6)
        features['MDVP:PPQ'] = jitter_std / np.sqrt(len(zero_crossings) + 1e-6)
        features['Jitter:DDP'] = jitter_std * 3

        # Shimmer-related features
        amplitude = librosa.amplitude_to_db(
            np.abs(y),
            ref=np.max
        )

        shimmer_std = np.std(amplitude)
        shimmer_mean = np.mean(amplitude)

        features['MDVP:Shimmer'] = (
            shimmer_std / shimmer_mean
            if shimmer_mean != 0
            else np.nan
        )

        features['MDVP:Shimmer(dB)'] = shimmer_std
        features['Shimmer:APQ3'] = shimmer_std / 3
        features['Shimmer:APQ5'] = shimmer_std / 5
        features['MDVP:APQ'] = shimmer_std / len(amplitude)
        features['Shimmer:DDA'] = shimmer_std * 3

        # Noise-to-Harmonic Ratio
        harmonic, percussive = librosa.effects.hpss(y)

        features['NHR'] = (
            np.mean(percussive) /
            (np.mean(harmonic) + 1e-6)
        )

        features['HNR'] = (
            np.mean(harmonic) /
            (np.mean(percussive) + 1e-6)
        )

        # Nonlinear dynamic features
        features['RPDE'] = (
            entropy(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features['DFA'] = librosa.feature.rms(y=y).mean()

        # Spread and PPE
        features['spread1'] = (
            np.std(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features['spread2'] = (
            np.var(pitches)
            if len(pitches) > 0
            else np.nan
        )

        features['D2'] = (
            np.percentile(pitches, 99)
            if len(pitches) > 0
            else np.nan
        )

        features['PPE'] = (
            np.mean(
                np.abs(
                    pitches - np.mean(pitches)
                )
            )
            if len(pitches) > 0
            else np.nan
        )

        return features

    except Exception as e:
        print(f"Feature extraction failed for {file_path}: {e}")
        return None


# ---------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------

def create_dataset(audio_folder, output_csv, label):
    data = []

    for file_name in os.listdir(audio_folder):

        if file_name.endswith((".wav", ".mp3", ".ogg")):

            file_path = os.path.join(
                audio_folder,
                file_name
            )

            features = extract_features(file_path)

            if features is not None:

                features['status'] = label
                features['File_Name'] = file_name

                data.append(features)

    df = pd.DataFrame(data)

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Dataset for label={label} "
        f"saved to {output_csv}"
    )


# ---------------------------------------------------
# PARKINSON'S DATASET
# ---------------------------------------------------

pd_audio_folder = os.path.join(
    os.getcwd(),
    "PD_AH"
)

create_dataset(
    pd_audio_folder,
    "parkinsons_dataset.csv",
    label=1
)


# ---------------------------------------------------
# HEALTHY CONTROL DATASET
# ---------------------------------------------------

hc_audio_folder = os.path.join(
    os.getcwd(),
    "HC_AH"
)

create_dataset(
    hc_audio_folder,
    "healthy_dataset.csv",
    label=0
)


# ---------------------------------------------------
# COMBINE DATASETS
# ---------------------------------------------------

healthy_df = pd.read_csv(
    "healthy_dataset.csv"
)

parkinsons_df = pd.read_csv(
    "parkinsons_dataset.csv"
)

combined_df = pd.concat(
    [healthy_df, parkinsons_df],
    ignore_index=True
)

combined_df.to_csv(
    "all_dataset.csv",
    index=False
)

print(
    "Combined dataset saved as all_dataset.csv"
)