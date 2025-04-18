import numpy as np
import librosa
import opensmile
import os

# ====== LIBROSA-BASED MFCC EXTRACTION ======
def extract_mfcc_librosa(file_path: str, n_mfcc: int = 89, sr: int = 16000) -> np.ndarray:
    """
    Extracts mean MFCC features from an audio file using Librosa.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients to extract.
        sr (int): Sample rate for loading the audio.

    Returns:
        np.ndarray: Array of shape (n_mfcc,) containing the mean MFCCs.
    """
    try:
        signal, sample_rate = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"[ERROR] Failed to extract MFCCs from {file_path}: {e}")
        return None


# ====== OPENSMILE FEATURE EXTRACTION ======
def init_opensmile_extractor():
    """
    Initializes an OpenSMILE feature extractor.

    Returns:
        opensmile.Smile: Configured OpenSMILE object.
    """
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals
    )


def extract_opensmile_features(file_path: str, smile_extractor=None) -> np.ndarray:
    """
    Extracts audio features using OpenSMILE.

    Args:
        file_path (str): Path to the audio file.
        smile_extractor (opensmile.Smile, optional): Pre-initialized Smile object.

    Returns:
        np.ndarray: Feature vector extracted by OpenSMILE.
    """
    try:
        if smile_extractor is None:
            smile_extractor = init_opensmile_extractor()

        features_df = smile_extractor.process_file(file_path)
        return features_df.values.flatten()
    except Exception as e:
        print(f"[ERROR] OpenSMILE failed on {file_path}: {e}")
        return None


# ====== BATCH FEATURE EXTRACTION ======
def extract_features_from_directory(directory_path, method="librosa", **kwargs):
    """
    Extract features from all .wav files in a directory using the specified method.

    Args:
        directory_path (str): Path to the directory containing .wav files.
        method (str): Feature extraction method ('librosa' or 'opensmile').

    Returns:
        List[np.ndarray], List[str]: Feature vectors and corresponding filenames.
    """
    features = []
    file_list = []

    if method == "opensmile":
        smile_extractor = init_opensmile_extractor()
    else:
        smile_extractor = None

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".wav"):
            path = os.path.join(directory_path, filename)
            if method == "librosa":
                feat = extract_mfcc_librosa(path, **kwargs)
            elif method == "opensmile":
                feat = extract_opensmile_features(path, smile_extractor)
            else:
                raise ValueError(f"Unknown method: {method}")

            if feat is not None:
                features.append(feat)
                file_list.append(filename)

    return features, file_list
