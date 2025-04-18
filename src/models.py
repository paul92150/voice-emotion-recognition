from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# ====== SVM MODEL TRAINING ======
def train_svm(X_train, y_train, kernel='linear', C=1.0, degree=3, gamma='scale'):
    """
    Train an SVM classifier.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        kernel (str): Kernel type ('linear', 'rbf', 'poly', etc.).
        C (float): Regularization parameter.
        degree (int): Degree for polynomial kernel.
        gamma (str or float): Kernel coefficient.

    Returns:
        model: Trained SVM model.
    """
    model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, probability=True)
    model.fit(X_train, y_train)
    return model

# ====== LOGISTIC REGRESSION MODEL TRAINING ======
def train_logreg(X_train, y_train, C=1.0, max_iter=1000):
    """
    Train a logistic regression classifier.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        C (float): Inverse of regularization strength.
        max_iter (int): Maximum iterations.

    Returns:
        model: Trained logistic regression model.
    """
    model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
    model.fit(X_train, y_train)
    return model

# ====== GRID SEARCH FOR SVM ======
def grid_search_svm(X_train, y_train, param_grid=None, cv=3):
    """
    Perform GridSearchCV for SVM hyperparameter tuning.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        param_grid (dict): Grid of parameters (default uses common values).
        cv (int): Number of cross-validation folds.

    Returns:
        best_model: Best estimator from grid search.
        best_params: Best parameters found.
    """
    if param_grid is None:
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4]
        }

    grid = GridSearchCV(SVC(probability=True), param_grid, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"[GridSearch] Best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_
