import numpy as np
import pandas as pd
import pickle
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# Load processed data
# -----------------------------
train_data = pd.read_csv("./data/processed/train_processed.csv")
test_data = pd.read_csv("./data/processed/test_processed.csv")

train_data.fillna("", inplace=True)
test_data.fillna("", inplace=True)

X_train = train_data["tweet"].values
y_train = train_data["label"].values

X_test = test_data["tweet"].values
y_test = test_data["label"].values

# -----------------------------
# Load parameters
# -----------------------------
params = yaml.safe_load(open("params.yaml"))

vectorizer = CountVectorizer(
    max_features=params["vectorizer"]["max_features"],
    ngram_range=(
        params["vectorizer"]["ngram_min"],
        params["vectorizer"]["ngram_max"]
    ),
    min_df=2
)

# -----------------------------
# Vectorization
# -----------------------------
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# -----------------------------
# Save artifacts (NUMPY only)
# -----------------------------
os.makedirs("data/features", exist_ok=True)

np.save("data/features/X_train.npy", X_train_bow.toarray())
np.save("data/features/y_train.npy", y_train)

np.save("data/features/X_test.npy", X_test_bow.toarray())
np.save("data/features/y_test.npy", y_test)

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
