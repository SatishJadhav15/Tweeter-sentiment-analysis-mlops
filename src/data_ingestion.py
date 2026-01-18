import pandas as pd
import os
from sklearn.model_selection import train_test_split


# Load local CSV (root-level)
def load_data(csv_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}")

        df = pd.read_csv(csv_path)
        return df

    except FileNotFoundError as e:
        print("File error while loading data")
        print(e)
        raise
    except Exception as e:
        print("Unexpected error while loading data")
        print(e)
        raise


# Save processed data
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: str):
    try:
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        train_df.to_csv(os.path.join(raw_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_dir, "test.csv"), index=False)

    except Exception as e:
        print("Error while saving data")
        print(e)
        raise

 

def main():
    try:
        # project root (ML-pipeline-DVC)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # input CSV at project root
        input_csv = os.path.join(project_root,'data', "tweets.csv")
        # output data directory (outside src)
        data_dir = os.path.join(project_root, "data")
        df = load_data(input_csv)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )

        save_data(train_df, test_df, data_dir)

        print("Data ingestion completed successfully")

    except Exception:
        print("Data ingestion failed")
        raise


if __name__ == "__main__":
    main()
