import os
import kagglehub
from tqdm import tqdm

def download_and_extract_nih_dataset(output_folder):
    """
    Downloads and extracts the NIH Chest X-ray dataset using KaggleHub.
    """
    print("Step 1: Initiating download...")

    # Use KaggleHub to download the dataset
    path = kagglehub.dataset_download("nih-chest-xrays/data")

    print("Step 2: Download complete.")
    print("Path to dataset files (temporary):", path)

    # Move files to the specified output folder
    os.makedirs(output_folder, exist_ok=True)

    # Count total files for progress bar
    total_files = sum([len(files) for _, _, files in os.walk(path)])

    with tqdm(total=total_files, desc="Moving files", unit="file") as progress_bar:
        for root, dirs, files in os.walk(path):
            for file in files:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_folder, file)
                os.rename(source_path, destination_path)
                progress_bar.update(1)

    print(f"Step 3: Dataset files moved to {output_folder}.")

if __name__ == "__main__":
    # Specify the output directory
    output_directory = "/scratch/torch-env/nih-chest-xray"

    # Download and extract the dataset
    download_and_extract_nih_dataset(output_directory)
