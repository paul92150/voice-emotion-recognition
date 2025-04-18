import pandas as pd
from sklearn.metrics import classification_report
from tpot.tpot_estimator import TPOTClassifier
import h2o
from h2o.automl import H2OAutoML
from joblib import parallel_backend 

# ====== TPOT AUTOML ======
def run_tpot_automl(X_train, y_train, X_test, y_test, label_encoder, generations=5, population_size=30):
    """
    Run TPOT AutoML safely with forced local backend (no Dask).
    """
    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        random_state=42,
        n_jobs=1,
    )

    print("[TPOT] Starting TPOT fit...")

    with parallel_backend('loky'): 
        tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)

    print("[TPOT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("best pipeline:", tpot.fitted_pipeline_)


    return tpot.fitted_pipeline_


# ====== H2O AUTOML ======
def run_h2o_automl(X_train, y_train, X_test, y_test, max_runtime=600):
    """
    Train and evaluate an AutoML model using H2O.

    Args:
        X_train, y_train, X_test, y_test: Training and testing data.
        max_runtime (int): Max runtime in seconds.

    Returns:
        model: Best model found by H2O AutoML.
    """
    h2o.init()

    # Prepare H2OFrame
    train_data = pd.DataFrame(X_train)
    train_data['label'] = y_train
    test_data = pd.DataFrame(X_test)
    test_data['label'] = y_test

    train_h2o = h2o.H2OFrame(train_data)
    test_h2o = h2o.H2OFrame(test_data)

    train_h2o['label'] = train_h2o['label'].asfactor()
    test_h2o['label'] = test_h2o['label'].asfactor()

    features = train_h2o.columns[:-1]
    target = 'label'

    # Run AutoML
    aml = H2OAutoML(max_runtime_secs=max_runtime, seed=42)
    aml.train(x=features, y=target, training_frame=train_h2o)

    # Show leaderboard
    print(aml.leaderboard)

    # Evaluate on test set
    predictions = aml.leader.predict(test_h2o).as_data_frame()
    true_labels = test_h2o["label"].as_data_frame()["label"]
    predicted_labels = predictions["predict"]

    accuracy = (true_labels == predicted_labels).sum() / len(true_labels)
    print(f"[H2O] Test Accuracy: {accuracy:.2f}")

    print("[H2O] Confusion Matrix:")
    print(pd.crosstab(true_labels, predicted_labels, rownames=["Actual"], colnames=["Predicted"]))

    return aml.leader
