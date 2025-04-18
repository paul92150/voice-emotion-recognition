import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# ====== CALCULATE MI FOR FIXED FEATURES ======
def select_top_features_mi(X, y, threshold=0.04):
    """
    Select features with Mutual Information (MI) above a threshold.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Encoded labels.
        threshold (float): Minimum MI score to retain a feature.

    Returns:
        (X_selected, selected_indices, mi_scores): Filtered features, their indices, and MI scores.
    """
    mi_scores = mutual_info_classif(X, y, discrete_features=False)
    selected_indices = np.where(mi_scores > threshold)[0]
    X_selected = X[:, selected_indices]
    return X_selected, selected_indices, mi_scores

# ====== EVALUATE MI ACROSS MFCC RANGES ======
def evaluate_mi_across_mfcc_ranges(mfcc_data_by_count: dict, y, plot=True):
    """
    Evaluate average MI scores for multiple MFCC counts.

    Args:
        mfcc_data_by_count (dict): Dict where keys are n_mfcc values and values are feature lists.
        y (list or np.ndarray): Encoded labels.
        plot (bool): Whether to show a plot of MI scores vs. n_mfcc.

    Returns:
        dict: Mapping from n_mfcc to average MI score.
    """
    mi_scores = {}
    for n_mfcc, features_list in mfcc_data_by_count.items():
        X = np.array(features_list)
        if X.shape[0] == len(y):
            score = mutual_info_classif(X, y, discrete_features=False)
            mi_scores[n_mfcc] = np.mean(score)
        else:
            print(f"[WARN] Skipping n_mfcc={n_mfcc}: length mismatch (X={len(X)}, y={len(y)})")

    if plot and mi_scores:
        x_vals = sorted(mi_scores.keys())
        y_vals = [mi_scores[k] for k in x_vals]
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel("n_mfcc")
        plt.ylabel("Avg. Mutual Information Score")
        plt.title("Mutual Information vs. Number of MFCCs")
        plt.grid(True)
        plt.show()

    return mi_scores
