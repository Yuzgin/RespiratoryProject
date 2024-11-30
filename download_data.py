import kagglehub

def download_and_reference_dataset():
    """
    Downloads the NIH Chest X-ray dataset using KaggleHub without moving to a specific output directory.
    """
    print("Step 1: Initiating download...")

    # Use KaggleHub to download the dataset
    path = kagglehub.dataset_download("nih-chest-xrays/data")

    print("Step 2: Download complete.")
    print(f"Dataset is ready for use at {path}.")

if __name__ == "__main__":
    # Download the dataset directly
    download_and_reference_dataset()
