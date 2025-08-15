import os
import shutil
from zipfile import ZipFile

import requests
from tqdm import tqdm


def download_file(url, save_path):
    """
    Downloads a file from a given URL and saves it to a specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.
    """
    try:
        response = requests.get(url, stream=True)  # Use stream=True for large files
        response.raise_for_status()  # Raise an error for bad responses

        total_size = int(response.headers.get("content-length", 0))

        print("Downloading file...")
        with open(save_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=save_path) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"File downloaded successfully: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")


def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to a specified directory.

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


data_folder = "data"

os.makedirs(data_folder, exist_ok=True)

file_url = "https://www.kaggle.com/api/v1/datasets/download/utkarshsaxenadn/fruits-classification"
local_save_path = f"{data_folder}/fruits-classification.zip"

download_file(file_url, local_save_path)

extract_to_path = f"{data_folder}/fruits-dataset-tmp"
final_path = f"{data_folder}/fruits-dataset"

if os.path.exists(extract_to_path) and os.path.isdir(extract_to_path):
    shutil.rmtree(extract_to_path)

if os.path.exists(final_path) and os.path.isdir(final_path):
    shutil.rmtree(final_path)

extract_zip(local_save_path, extract_to_path)
os.remove(local_save_path)

shutil.move(f"{extract_to_path}/Fruits Classification", final_path)
shutil.rmtree(extract_to_path)
shutil.move(f"{final_path}/valid", f"{final_path}/val")
