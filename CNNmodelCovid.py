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

class CovidXrayDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.data = self.load_data()
        self.label_map = self.create_label_map()

    def load_data(self):
        categories = ['COVID', 'NORMAL', 'Viral Pneumonia']
        data = []

        for category in categories:
            folder = os.path.join(self.images_folder, category)
            if not os.path.exists(folder):
                continue
            files = glob(os.path.join(folder, '*.png')) + glob(os.path.join(folder, '*.jpg'))
            for file in files:
                data.append((file, category))

        df = pd.DataFrame(data, columns=['path', 'label'])
        print(f"Loaded dataset with {len(df)} images.")
        return df

    def create_label_map(self):
        labels = sorted(self.data['label'].unique())
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['path']
        image = Image.open(img_path).convert('RGB')

        label = row['label']
        target = torch.zeros(len(self.label_map))
        target[self.label_map[label]] = 1

        if self.transform:
            image = self.transform(image)

        return image, target

def main():
    data_folder = "../covid19-dataset/COVID-19_Radiography_Dataset"
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CovidXrayDataset(images_folder=data_folder, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    batch_size = int(input("Enter batch size: "))
    num_workers = int(input("Enter number of workers: "))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load ResNet50 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = len(dataset.label_map)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Use 4 GPUs explicitly
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    print("Using GPUs 0, 1, 2, 3")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "covid_resnet50.pth")
    print("Model saved to covid_resnet50.pth")

    # Evaluation
    model.eval()
    all_targets = []
    all_outputs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    try:
        auc_scores = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
                auc_scores.append(auc)
        mean_auc = np.mean(auc_scores)
        print(f"Mean AUC score: {mean_auc:.4f}")
    except Exception as e:
        print(f"Failed to calculate AUC: {e}")

if __name__ == "__main__":
    main()
