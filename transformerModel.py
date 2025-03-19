import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from glob import glob
from tqdm import tqdm
import subprocess
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

def get_best_gpus(num_gpus=4):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_memories = [(int(index), int(memory.strip())) for index, memory in 
                    (line.split(',') for line in result.stdout.splitlines())]
    return [x[0] for x in sorted(gpu_memories, key=lambda x: x[1], reverse=True)[:num_gpus]]

class ChestXrayDataset(Dataset):
    def __init__(self, images_folder, csv_file, transform=None, subset_list=None):
        self.images_folder = images_folder
        self.transform = transform
        self.data = self.load_data(csv_file, subset_list)
        self.label_map = self.create_label_map()
    
    def load_data(self, csv_file, subset_list):
        df = pd.read_csv(csv_file)
        if subset_list:
            with open(subset_list, 'r') as f:
                image_list = [line.strip() for line in f.readlines()]
            df = df[df['Image Index'].isin(image_list)]
        df['path'] = df['Image Index'].map(self.get_image_paths())
        df.dropna(subset=['path'], inplace=True)
        return df
    
    def get_image_paths(self):
        search_path = os.path.join(self.images_folder, 'images_*/images', '*.png')
        return {os.path.basename(x): x for x in glob(search_path)}
    
    def create_label_map(self):
        labels = set().union(*[item.split('|') for item in self.data['Finding Labels']])
        return {label: idx for idx, label in enumerate(sorted(labels))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        target = torch.zeros(len(self.label_map))
        for label in row['Finding Labels'].split('|'):
            if label in self.label_map:
                target[self.label_map[label]] = 1
        return self.transform(image) if self.transform else image, target

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
    
    all_targets, all_outputs = np.vstack(all_targets), np.vstack(all_outputs)
    auc = roc_auc_score(all_targets, all_outputs, average='macro')
    acc = correct / total
    return total_loss / len(loader), auc, acc

def main():
    data_folder = "/shared/storage/cs/studentscratch/ay841/nih-chest-xrays"
    csv_file = os.path.join(data_folder, "Data_Entry_2017.csv")
    train_val_list = os.path.join(data_folder, "train_val_list.txt")

    with open(train_val_list, 'r') as f:
        all_files = [line.strip() for line in f.readlines()]

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_list = os.path.join(data_folder, "train_list.txt")
    val_list = os.path.join(data_folder, "val_list.txt")

    with open(train_list, 'w') as f:
        f.write("\n".join(train_files))

    with open(val_list, 'w') as f:
        f.write("\n".join(val_files))

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ChestXrayDataset(data_folder, csv_file, transform, train_list)
    val_dataset = ChestXrayDataset(data_folder, csv_file, transform, val_list)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    num_classes = len(train_dataset.label_map)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    best_gpus = get_best_gpus(num_gpus=4)
    device = torch.device(f"cuda:{best_gpus[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=best_gpus)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 40
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
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_auc, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auc:.4f} | Val Acc: {val_acc:.4f}')

if __name__ == "__main__":
    main()
