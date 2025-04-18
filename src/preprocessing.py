import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter

# ====== NORMALIZATION ======
def normalize_features(X: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize feature array using specified method.

    Args:
        X (np.ndarray): Feature matrix.
        method (str): Normalization method. Currently supports 'minmax'.

    Returns:
        np.ndarray: Normalized feature matrix.
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    else:
        raise ValueError(f"Normalization method '{method}' is not supported.")

# ====== PADDING ======
def pad_sequences(sequences, max_length=None, padding_value=-1.0):
    """
    Pad sequences to the same length.

    Args:
        sequences (list of np.ndarray): List of 1D feature arrays.
        max_length (int): Desired sequence length (if None, use longest).
        padding_value (float): Value to pad with.

    Returns:
        np.ndarray: Padded 2D array (n_samples, max_length).
    """
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    padded = np.full((len(sequences), max_length), padding_value)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
    return padded

# ====== LABEL MAPPING FROM FILENAME ======
def map_label_from_filename(filename: str, position: int, mapping: dict) -> str:
    """
    Extract label from filename using a predefined character-to-label mapping.

    Args:
        filename (str): Audio filename.
        position (int): Index to extract label character from filename.
        mapping (dict): Dictionary mapping characters to emotion labels.

    Returns:
        str: Mapped label or None if not found.
    """
    if len(filename) > position:
        code = filename[position]
        return mapping.get(code)
    return None

# ====== CLASS FILTERING ======
def filter_classes(X, y, min_samples=2):
    """
    Remove classes with fewer than min_samples instances.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label array.
        min_samples (int): Minimum number of samples per class.

    Returns:
        (X_filtered, y_filtered): Filtered data and labels.
    """
    counts = Counter(y)
    valid_labels = [label for label, count in counts.items() if count >= min_samples]
    indices = [i for i, label in enumerate(y) if label in valid_labels]
    return X[indices], np.array(y)[indices]

# ====== LABEL ENCODING ======
def encode_labels(y: list):
    """
    Encode class labels to integers.

    Args:
        y (list): List of string labels.

    Returns:
        (np.ndarray, LabelEncoder): Encoded labels and fitted encoder.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

# Function to map the label from the filename
def map_label_from_filename(filename, label_position, emotion_mapping):
    """
    Map the emotion label from the filename.
    Assumes the emotion label is at the `label_position` in the filename.
    """
    emotion_label = filename[label_position]  # Extract the emotion label from filename
    return emotion_mapping.get(emotion_label, None)  # Return the mapped label or None if not found

