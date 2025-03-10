import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet50, ResNet50_Weights
from glob import glob
from tqdm import tqdm
import subprocess
from sklearn.metrics import roc_auc_score

def get_best_gpus(num_gpus=4):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_memories = [(int(index), int(memory.strip())) for index, memory in 
                    (line.split(',') for line in result.stdout.splitlines())]
    sorted_gpus = sorted(gpu_memories, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_gpus[:num_gpus]]

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNTransformer, self).__init__()
        
        # Load ResNet-50 as CNN feature extractor
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove fully connected layers
        
        # Load Vision Transformer
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.transformer = vit.encoder
        self.cls_token = vit.class_token
        
        # Classifier
        self.fc = nn.Linear(768, num_classes)  # ViT output feature size
        
    def forward(self, x):
        cnn_features = self.cnn_backbone(x)  # Extract CNN features
        cnn_features = cnn_features.flatten(2).mean(dim=2)  # Global Average Pooling
        
        # Pass through ViT encoder
        vit_input = torch.cat([self.cls_token.expand(x.size(0), -1, -1), cnn_features.unsqueeze(1)], dim=1)
        vit_output = self.transformer(vit_input)
        
        # Classification
        output = self.fc(vit_output[:, 0])
        return output

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

    num_classes = len(train_dataset.label_map)
    model = HybridCNNTransformer(num_classes)

    best_gpus = get_best_gpus(num_gpus=4)
    print(f"Using GPUs: {best_gpus}")
    device = torch.device(f"cuda:{best_gpus[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = nn.DataParallel(model, device_ids=best_gpus)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    torch.cuda.empty_cache()

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), "hybrid_model.pth")
    print("Model weights saved to hybrid_model.pth")

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
    all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()

    auc_scores = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
            auc_scores.append(auc)
        except ValueError:
            print(f"Skipping AUC calculation for class {i} due to only one label present.")
            auc_scores.append(None)

    valid_auc_scores = [auc for auc in auc_scores if auc is not None]
    mean_auc = sum(valid_auc_scores) / len(valid_auc_scores) if valid_auc_scores else 0

    print(f"Per-Class AUC Scores: {auc_scores}")
    print(f"Mean AUC Score: {mean_auc:.4f}")

if __name__ == "__main__":
    main()
