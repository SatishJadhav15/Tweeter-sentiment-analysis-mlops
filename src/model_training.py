import numpy as np
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------
# Load training data (NUMPY, not CSV)
# -----------------------------
X_train = np.load("data/features/X_train.npy")
y_train = np.load("data/features/y_train.npy")

# -----------------------------
# Load parameters
# -----------------------------
params = yaml.safe_load(open("params.yaml"))
n_estimators = params["model"]["n_estimators"]

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment("tweet-sentiment-dvc")

with mlflow.start_run():
    # log parameter
    mlflow.log_param("n_estimators", n_estimators)

    # -------------------------
    # Train model
    # -------------------------
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    # -------------------------
    # Save model locally (for DVC + Streamlit)
    # -------------------------
    pickle.dump(clf, open("model.pkl", "wb"))

    # -------------------------
    # Log model to MLflow
    # -------------------------
    mlflow.sklearn.log_model(clf, "model")
