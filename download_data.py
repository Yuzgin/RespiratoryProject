import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

def download_and_extract_nih_dataset(output_folder):
    """
    Downloads and extracts the NIH Chest X-ray dataset from the official NIH website.
    """
    # URL of the NIH dataset zip file
    dataset_url = "https://nihcc.app.box.com/v/ChestXray-NIHCC"
    zip_file_name = "chest_xray_dataset.zip"
    zip_file_path = os.path.join(output_folder, zip_file_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the dataset has already been downloaded
    if not os.path.exists(zip_file_path):
        print("Step 1: Checking and initiating download...")
        print(f"Downloading dataset from {dataset_url}...")
        
        # Initiate download with progress bar
        with requests.get(dataset_url, stream=True) as response:
            response.raise_for_status()  # Raise an error for HTTP issues
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_file_path, "wb") as zip_file, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):
                    zip_file.write(chunk)
                    progress_bar.update(len(chunk))
        print("Step 2: Download complete.")
    else:
        print("Dataset already downloaded. Skipping download.")

    # Extract the dataset
    print("Step 3: Extracting dataset...")
    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    print("Step 4: Dataset extraction complete.")

if __name__ == "__main__":
    # Specify the output directory
    output_directory = "data"

    # Download and extract the dataset
    download_and_extract_nih_dataset(output_directory)
