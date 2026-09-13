"""
Data ingestion module for Mayo Clinic Primary Biliary Cirrhosis (PBC) dataset.
Provides local caching and fallback auto-downloading from verified repositories.
"""

import logging
from pathlib import Path
from typing import Optional
import urllib.request
import pandas as pd

from config.config import RAW_DATA_PATH, REMOTE_DATASET_URL

logger = logging.getLogger(__name__)


def load_raw_data(
    filepath: Optional[Path] = None,
    auto_download: bool = True
) -> pd.DataFrame:
    """
    Load raw Mayo Clinic cirrhosis/PBC dataset from local disk or verified remote source.

    Args:
        filepath: Custom path to raw dataset CSV file. Defaults to config.RAW_DATA_PATH.
        auto_download: If True and local file is absent, download from canonical remote source.

    Returns:
        pd.DataFrame containing raw patient records.

    Raises:
        FileNotFoundError: If local file does not exist and auto_download is False.
        ValueError: If loaded dataset is empty.
    """
    target_path = Path(filepath) if filepath else RAW_DATA_PATH

    if not target_path.exists():
        if auto_download:
            logger.info("Dataset not found at %s. Downloading canonical Mayo Clinic PBC dataset...", target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                req = urllib.request.Request(
                    REMOTE_DATASET_URL,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=15) as response, open(target_path, "wb") as out_file:
                    out_file.write(response.read())
                logger.info("Successfully saved dataset to %s", target_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download Mayo Clinic PBC dataset from {REMOTE_DATASET_URL}. "
                    f"Please place cirrhosis.csv into {target_path.parent}. Error: {exc}"
                ) from exc
        else:
            raise FileNotFoundError(
                f"Dataset file not found at {target_path}. Set auto_download=True or provide cirrhosis.csv."
            )

    df = pd.read_csv(target_path)
    if df.empty:
        raise ValueError(f"Loaded dataset at {target_path} is empty.")

    logger.info("Loaded dataset successfully with shape %s from %s", df.shape, target_path)
    return df


def save_raw_data(df: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """Save raw dataframe to disk."""
    target_path = Path(filepath) if filepath else RAW_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info("Saved raw dataset to %s", target_path)
    return target_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_raw_data()
    print("Dataset preview:")
    print(data.head())
