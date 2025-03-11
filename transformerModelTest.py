import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
import pandas as pd
from glob import glob

class ChestXrayDataset(torch.utils.data.Dataset):
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
        image_paths = {os.path.basename(x): x for x in glob(search_path)}
        return image_paths

    def create_label_map(self):
        labels = set()
        for item in self.data['Finding Labels']:
            labels.update(item.split('|'))
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['path']
        image = Image.open(img_path).convert('RGB')
        labels = row['Finding Labels'].split('|')
        target = torch.zeros(len(self.label_map))
        for label in labels:
            if label in self.label_map:
                target[self.label_map[label]] = 1
        if self.transform:
            image = self.transform(image)
        return image, target


def evaluate():
    data_folder = "/shared/storage/cs/studentscratch/ay841/nih-chest-xrays"
    csv_file = os.path.join(data_folder, "Data_Entry_2017.csv")
    test_list = os.path.join(data_folder, "test_list.txt")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = ChestXrayDataset(data_folder, csv_file, transform_test, test_list)

    # Ask for batch size and number of workers
    batch_size = int(input("Enter batch size: "))
    num_workers = int(input("Enter number of workers: "))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    num_classes = len(test_dataset.label_map)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Load weights
    model.load_state_dict(torch.load("vit_b16_trained.pth", strict=False))

    # Set up 4 GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use 4 GPUs explicitly
    print("Using GPUs 0, 1, 2, 3")

    model = model.to(device)
    model.eval()

    all_targets = []
    all_outputs = []
    correct = 0
    total = 0

    print("Evaluating model...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            total += targets.numel()
            correct += (predicted == targets).sum().item()

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    # Compute accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Compute AUC
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    try:
        auc_scores = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
                auc_scores.append(auc)
        mean_auc = np.mean(auc_scores)
        print(f'Test AUC Score: {mean_auc:.4f}')
    except Exception as e:
        print(f"Failed to calculate AUC: {e}")

if __name__ == "__main__":
    evaluate()

