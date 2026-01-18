import numpy as np
import pandas as pd
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------
# Load training data
# -----------------------------
train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

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
    # Save model locally (for DVC)
    # -------------------------
    pickle.dump(clf, open('model.pkl', 'wb'))

    # -------------------------
    # Log model to MLflow
    # -------------------------
    mlflow.sklearn.log_model(clf, "model")
