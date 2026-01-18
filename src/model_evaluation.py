import numpy as np
import pandas as pd
import pickle
import json
import mlflow

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# -----------------------------
# Load model and test data
# -----------------------------
clf = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv('./data/features/test_bow.csv')

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# -----------------------------
# Predictions
# -----------------------------
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "auc": auc
}

# -----------------------------
# Save metrics for DVC
# -----------------------------
with open("metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)

# -----------------------------
# Log metrics to MLflow
# -----------------------------
mlflow.set_experiment("tweet-sentiment-dvc")

with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("auc", auc)
