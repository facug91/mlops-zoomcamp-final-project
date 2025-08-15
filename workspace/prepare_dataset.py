import os
import shutil
from zipfile import ZipFile

import requests
from tqdm import tqdm

DATA_FOLDER = "data"
ZIP_FILENAME = "fruits-classification.zip"
ZIP_PATH = os.path.join(DATA_FOLDER, ZIP_FILENAME)
EXTRACT_TMP_PATH = os.path.join(DATA_FOLDER, "fruits-dataset-tmp")
FINAL_PATH = os.path.join(DATA_FOLDER, "fruits-dataset")

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/utkarshsaxenadn/fruits-classification"


def download_file(url: str, save_path: str) -> None:
    """
    Download a file from a given URL and save it to the specified path.
    Shows a progress bar for large files.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        print(f"Downloading from {url} ...")
        with (
            open(save_path, "wb") as f,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(save_path)) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"File downloaded successfully: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        raise


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip file to a specified directory.

    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where the contents will be extracted.
    """
    try:
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted files to: {extract_to}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        raise


def clean_and_prepare_dirs(*dirs: str) -> None:
    """Remove directories if they already exist."""
    for d in dirs:
        if os.path.exists(d) and os.path.isdir(d):
            shutil.rmtree(d)


if __name__ == "__main__":
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Step 1: Download dataset
    download_file(DATASET_URL, ZIP_PATH)

    # Step 2: Clean previous folders
    clean_and_prepare_dirs(EXTRACT_TMP_PATH, FINAL_PATH)

    # Step 3: Extract dataset
    extract_zip(ZIP_PATH, EXTRACT_TMP_PATH)
    os.remove(ZIP_PATH)

    # Step 4: Move to final structure
    shutil.move(os.path.join(EXTRACT_TMP_PATH, "Fruits Classification"), FINAL_PATH)
    shutil.rmtree(EXTRACT_TMP_PATH)

    # Rename "valid" to "val"
    shutil.move(os.path.join(FINAL_PATH, "valid"), os.path.join(FINAL_PATH, "val"))

    print("Dataset prepared successfully.")
