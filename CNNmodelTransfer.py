import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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
        image_paths = {os.path.basename(x): x for x in glob(search_path)}
        return image_paths

    def create_label_map(self):
        labels = set()
        for item in self.data['Finding Labels']:
            labels.update(item.split('|'))
        labels = sorted(labels)
        return {label: idx for idx, label in enumerate(labels)}

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


def load_model(model_path, num_classes, device):
    model = models.resnet50(weights=None)  # No default weights since loading from file
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)

        # Remove 'module.' prefix if present
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Model file {model_path} not found. Starting from scratch.")
    
    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_outputs = []
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    try:
        auc_scores = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
                auc_scores.append(auc)
        mean_auc = np.mean(auc_scores)
        print(f'Mean AUC score: {mean_auc:.4f}')
    except Exception as e:
        print(f"Failed to calculate AUC: {e}")


def train_model(model, train_loader, device, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


def main():
    data_folder = "/shared/storage/cs/studentscratch/ay841/nih-chest-xrays"
    csv_file = os.path.join(data_folder, "Data_Entry_2017.csv")
    train_list = os.path.join(data_folder, "train_val_list.txt")
    test_list = os.path.join(data_folder, "test_list.txt")

    # Transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ChestXrayDataset(data_folder, csv_file, transform, train_list)
    test_dataset = ChestXrayDataset(data_folder, csv_file, transform, test_list)

    batch_size = int(input("Enter batch size: "))
    num_workers = int(input("Enter number of workers: "))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "resnet50_covid_trained.pth"
    num_classes = len(train_dataset.label_map)

    model = load_model(model_path, num_classes, device)

    # Force use of GPUs 0, 1, 2, 3 if available
    if torch.cuda.device_count() >= 4:
        print("Using GPUs 0, 1, 2, 3")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    else:
        print(f"Using {torch.cuda.device_count()} available GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Test the loaded model
    print("\n=== Evaluating Pre-trained Model ===")
    evaluate_model(model, test_loader, device)

    # Resume training
    print("\n=== Resuming Training ===")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_model(model, train_loader, device, criterion, optimizer, num_epochs)

    # Save the new model
    torch.save(model.state_dict(), "resnet50_covid_trained_updated.pth")
    print("Updated model saved to resnet50_covid_trained_updated.pth")

    # Test the updated model
    print("\n=== Evaluating Updated Model ===")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
