import os
import re
import string
import pandas as pd
import nltk
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
    except Exception as e:
        print("Error downloading NLTK resources")
        raise e

def lower_case(text: str) -> str:
    return text.lower()

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)

def removing_numbers(text: str) -> str:
    return "".join(char for char in text if not char.isdigit())

def removing_punctuations(text: str) -> str:
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def removing_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

def normalize_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    try:
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not found")

        df = df.copy()
        df[text_col] = df[text_col].astype(str)

        df[text_col] = (
            df[text_col]
            .apply(lower_case)
            .apply(remove_stop_words)
            .apply(removing_numbers)
            .apply(removing_punctuations)
            .apply(removing_urls)
            .apply(lemmatization)
            .apply(normalize_unicode)
        )
        df.dropna(subset=[text_col], inplace=True)
        return df
    except Exception as e:
        print("Error during text preprocessing")
        raise e

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    try:
        download_nltk_resources()
        # resolve project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # input paths
        train_path = os.path.join(project_root, "data", "raw", "train.csv")
        test_path = os.path.join(project_root, "data", "raw", "test.csv")
        # load data
        train_df = load_data(train_path)
        test_df = load_data(test_path)
        # preprocess
        train_processed = normalize_text(train_df, text_col="tweet")
        test_processed = normalize_text(test_df, text_col="tweet")
        # save output
        processed_dir = os.path.join(project_root, "data", "processed")
        save_data(train_processed, os.path.join(processed_dir, "train_processed.csv"))
        save_data(test_processed, os.path.join(processed_dir, "test_processed.csv"))
        print("Data preprocessing completed successfully")
    except Exception:
        print("Data preprocessing failed")
        raise

if __name__ == "__main__":
    main()
