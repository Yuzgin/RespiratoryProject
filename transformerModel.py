import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from glob import glob
from tqdm import tqdm
import subprocess
from sklearn.metrics import roc_auc_score  # Import for AUC calculation

# Function to get the GPUs with the most available memory using nvidia-smi
def get_best_gpus(num_gpus=4):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_memories = [(int(index), int(memory.strip())) for index, memory in 
                    (line.split(',') for line in result.stdout.splitlines())]
    sorted_gpus = sorted(gpu_memories, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_gpus[:num_gpus]]

def main():
    data_folder = "/shared/storage/cs/studentscratch/ay841/nih-chest-xrays"
    csv_file = os.path.join(data_folder, "Data_Entry_2017.csv")
    train_list = os.path.join(data_folder, "train_val_list.txt")
    test_list = os.path.join(data_folder, "test_list.txt")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ChestXrayDataset(data_folder, csv_file, transform_train, train_list)
    test_dataset = ChestXrayDataset(data_folder, csv_file, transform_test, test_list)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=32)

    # Load pretrained Vision Transformer
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Adjust the classifier head for multi-label classification
    num_classes = len(train_dataset.label_map)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Get the top 4 GPUs with the most available memory
    best_gpus = get_best_gpus(num_gpus=4)
    print(f"Using GPUs: {best_gpus}")

    device = torch.device(f"cuda:{best_gpus[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Wrap model with DataParallel for multi-GPU usage
    model = nn.DataParallel(model, device_ids=best_gpus)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    torch.cuda.empty_cache()

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Add progress bar with tqdm
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm progress bar
            loop.set_postfix(loss=loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), "vit_b16_trained.pth")
    print("Model weights saved to vit_b16_trained.pth")

    # **Evaluation Phase with AUC Calculation**
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())

    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    # Apply sigmoid to outputs to get probabilities
    all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()

    # Compute AUC for each class
    auc_scores = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
            auc_scores.append(auc)
        except ValueError:
            print(f"Skipping AUC calculation for class {i} due to only one label present.")
            auc_scores.append(None)  # Placeholder for failed AUC computation

    valid_auc_scores = [auc for auc in auc_scores if auc is not None]
    mean_auc = sum(valid_auc_scores) / len(valid_auc_scores) if valid_auc_scores else 0

    print(f"Per-Class AUC Scores: {auc_scores}")
    print(f"Mean AUC Score: {mean_auc:.4f}")

if __name__ == "__main__":
    main()
    
