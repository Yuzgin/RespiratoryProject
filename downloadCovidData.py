import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# Define the dataset name and output path
dataset_name = 'tawsifurrahman/covid19-radiography-database'
output_path = '../covid19-dataset'
zip_file = '../covid19-dataset.zip'

def download_dataset():
    print(f"Downloading dataset: {dataset_name}...")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset
    api.dataset_download_files(dataset_name, path='.', unzip=False)

    print(f"Downloaded to '{zip_file}'. Extracting...")

    # Extract with loading bar
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(total=len(file_list), desc="Extracting", unit="file") as pbar:
            for file in file_list:
                zip_ref.extract(file, output_path)
                pbar.update(1)
    
    # Clean up the zip file
    os.remove(zip_file)
    print(f"Dataset extracted to '{output_path}'")

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    download_dataset()

