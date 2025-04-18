import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from _setup_path import *
from src.features import extract_mfcc_librosa
from src.preprocessing import normalize_features, encode_labels
from src.preprocessing import map_label_from_filename

# === Configuration ===
AUDIO_DIR = "data/dataset1"
LABEL_POSITION = 5
N_MFCC_START = 1
N_MFCC_END = 89
STEP_SIZE = 2
MI_THRESHOLD = 0.12  # Set threshold for minimum mutual information score

EMOTION_MAPPING = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}

# === Feature Extraction ===
features = []
labels = []

for fname in os.listdir(AUDIO_DIR):
    if fname.lower().endswith(".wav"):
        label = map_label_from_filename(fname, LABEL_POSITION, EMOTION_MAPPING)
        if label is None:
            continue
        mfcc = extract_mfcc_librosa(os.path.join(AUDIO_DIR, fname), n_mfcc=N_MFCC_END)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

# === Preprocessing ===
X = normalize_features(X)
y_encoded, label_encoder = encode_labels(y)

# === Mutual Information Analysis ===
mi_scores = {}

for n_mfcc in range(N_MFCC_START, N_MFCC_END + 1, STEP_SIZE):
    # Select first n_mfcc features
    X_mfcc = X[:, :n_mfcc]

    # Calculate mutual information between features and target label
    mi_score = mutual_info_classif(X_mfcc, y_encoded)
    mi_scores[n_mfcc] = np.mean(mi_score)  # Take average MI score for n_mfcc features

# === Plot MI vs. n_mfcc ===
plt.plot(list(mi_scores.keys()), list(mi_scores.values()), marker='o')
plt.xlabel('Number of MFCCs')
plt.ylabel('Mutual Information Score')
plt.title('Mutual Information Score vs. Number of MFCCs')
plt.grid(True)
plt.show()

# === Apply MI Threshold ===
selected_mfcc_count = {}
for n_mfcc in range(N_MFCC_START, N_MFCC_END + 1, STEP_SIZE):
    # Select first n_mfcc features
    X_mfcc = X[:, :n_mfcc]

    # Calculate mutual information between features and target label
    mi_score = mutual_info_classif(X_mfcc, y_encoded)

    # Select features above threshold
    selected_features = np.where(mi_score > MI_THRESHOLD)[0]
    selected_mfcc_count[n_mfcc] = len(selected_features)

    print(f"MFCC {n_mfcc} - Selected features above threshold: {len(selected_features)}")

# === Plot Selected Features Above Threshold ===
plt.plot(list(selected_mfcc_count.keys()), list(selected_mfcc_count.values()), marker='o', color='r')
plt.xlabel('Number of MFCCs')
plt.ylabel('Number of Features Above Threshold')
plt.title(f'MFCC Count vs. Features Above Threshold ({MI_THRESHOLD})')
plt.grid(True)
plt.show()
