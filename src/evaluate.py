from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ====== CLASSIFICATION REPORT ======
def print_classification_report(model, X_test, y_test, label_encoder=None):
    """
    Print classification report with precision, recall, F1-score.

    Args:
        model: Trained model with .predict() method.
        X_test: Test features.
        y_test: Ground truth labels (encoded).
        label_encoder: Optional sklearn LabelEncoder (for readable class names).
    """
    y_pred = model.predict(X_test)

    target_names = label_encoder.classes_ if label_encoder else None

    print("=== Classification Report ===")
    print(classification_report(
        y_test,
        y_pred,
        target_names=target_names
    ))


# ====== CONFUSION MATRIX PLOTTING ======
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, figsize=(8, 6)):
    """
    Plot a confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of label names (optional).
        normalize: Whether to normalize counts to percentages.
        figsize: Size of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()
