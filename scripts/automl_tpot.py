from _setup_path import *
from src.features import extract_mfcc_librosa
from src.preprocessing import (
    normalize_features, encode_labels,
    map_label_from_filename, filter_classes
)
from src.automl import run_tpot_automl
from sklearn.model_selection import train_test_split
import numpy as np
import os

def main():
    # === CONFIG ===
    AUDIO_DIR = "data/dataset1"
    LABEL_POSITION = 5
    N_MFCC = 89
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    EMOTION_MAPPING = {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'anxiety',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }

    # === FEATURE EXTRACTION ===
    print("[INFO] Extracting features...")
    features, labels = [], []

    for fname in os.listdir(AUDIO_DIR):
        if fname.lower().endswith(".wav"):
            label = map_label_from_filename(fname, LABEL_POSITION, EMOTION_MAPPING)
            if label is None:
                continue
            mfcc = extract_mfcc_librosa(os.path.join(AUDIO_DIR, fname), n_mfcc=N_MFCC)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    X = normalize_features(X)
    X, y = filter_classes(X, y, min_samples=2)
    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
    )

    print("[INFO] Running TPOT AutoML...")
    run_tpot_automl(X_train, y_train, X_test, y_test, label_encoder)

if __name__ == '__main__':
    main()
