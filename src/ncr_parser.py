import pandas as pd
from pathlib import Path


def load_ncrs(filepath: str | Path = "data/sample_ncrs.csv") -> pd.DataFrame:
    """Load NCR data from CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file. Defaults to 'data/sample_ncrs.csv'.
    
    Returns:
        DataFrame containing NCR records.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


if __name__ == "__main__":
    df = load_ncrs()
    print(f"Loaded {len(df)} NCR records")
    print(df.info())
    print(df.head())
