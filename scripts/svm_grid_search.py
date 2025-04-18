import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump

from _setup_path import *
from src.features import extract_mfcc_librosa
from src.preprocessing import (
    normalize_features,
    encode_labels,
    filter_classes,
    map_label_from_filename,
)
from sklearn.preprocessing import StandardScaler

# === Configuration ===
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

# === Feature extraction ===
features = []
labels = []

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

# === Preprocessing ===
X = normalize_features(X)
X, y = filter_classes(X, y, min_samples=2)
y_encoded, label_encoder = encode_labels(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded
)

# === Pipeline ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# === Grid Search Parameters ===
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

print("[INFO] Starting Grid Search...")
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# === Results ===
print("\n[GRID SEARCH] Best parameters:", grid.best_params_)

y_pred = grid.predict(X_test)
print("\n[GRID SEARCH] Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Save best model ===
dump(grid.best_estimator_, "models/best_svm_model.joblib")
print("\n✅ Best SVM model saved to models/best_svm_model.joblib")
