import os
import numpy as np
from _setup_path import *
from src.features import extract_mfcc_librosa
from src.preprocessing import (
    normalize_features, encode_labels,
    map_label_from_filename, filter_classes
)
from src.models import train_logreg
from src.evaluate import print_classification_report, plot_confusion_matrix

from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
AUDIO_DIR = "data/dataset1"
LABEL_POSITION = 5
N_MFCC = 13
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
features = []
labels = []

for fname in os.listdir(AUDIO_DIR):
    if fname.lower().endswith(".wav"):
        label = map_label_from_filename(fname, LABEL_POSITION, EMOTION_MAPPING)
        if label is None:
            print(f"[WARN] Unrecognized label in {fname}. Skipping.")
            continue

        filepath = os.path.join(AUDIO_DIR, fname)
        mfcc = extract_mfcc_librosa(filepath, n_mfcc=N_MFCC)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)

if not features:
    print("[ERROR] No features extracted. Exiting.")
    exit()

X = np.array(features)
y = np.array(labels)

# === PREPROCESSING ===
X = normalize_features(X)
X, y = filter_classes(X, y, min_samples=2)
y_encoded, label_encoder = encode_labels(y)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
)

# === MODEL TRAINING ===
print("[INFO] Training Logistic Regression...")
model = train_logreg(X_train, y_train)

# === EVALUATION ===
print_classification_report(model, X_test, y_test, label_encoder)
plot_confusion_matrix(y_test, model.predict(X_test), labels=label_encoder.classes_)
