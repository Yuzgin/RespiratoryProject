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
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.data = self.load_data()
        self.label_map = self.create_label_map()

    def load_data(self):
        # Match actual folder names
        classes = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        data = []

        for label in classes:
            folder_path = os.path.join(self.images_folder, label, 'images')  # Search within 'images' folder
            if os.path.exists(folder_path):
                images = glob(os.path.join(folder_path, '*.png')) + \
                         glob(os.path.join(folder_path, '*.jpeg')) + \
                         glob(os.path.join(folder_path, '*.jpg')) + \
                         glob(os.path.join(folder_path, '*.JPG'))
                for img in images:
                    data.append({'path': img, 'label': label})
                print(f"{label} folder contains {len(images)} files")
            else:
                print(f"{label} folder does not exist")

        df = pd.DataFrame(data)
        print(f"Loaded dataset with {len(df)} images across {len(classes)} classes")
        return df

    def create_label_map(self):
        labels = sorted(self.data['label'].unique())
        label_map = {label.replace(' ', '_'): idx for idx, label in enumerate(labels)}
        print(f"Label map: {label_map}")
        return label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['path']
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        target = torch.zeros(len(self.label_map))
        target[self.label_map[label.replace(' ', '_')]] = 1

        if self.transform:
            image = self.transform(image)

        return image, target


def main():
    data_folder = "/shared/storage/cs/studentscratch/ay841/covid19-dataset/COVID-19_Radiography_Dataset"
    print(f"Dataset folder contents: {os.listdir(data_folder)}")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ChestXrayDataset(images_folder=data_folder, transform=transform_train)
    test_dataset = ChestXrayDataset(images_folder=data_folder, transform=transform_test)

    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    # Use GPU 0 and GPU 1 explicitly
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Ask user for batch size and number of workers
    batch_size = int(input("Enter batch size (for GPU 0): "))
    batch_size_gpu1 = batch_size * 2  # Double batch size for GPU 1
    num_workers = int(input("Enter number of workers: "))

    train_loader = DataLoader(train_dataset, batch_size=batch_size + batch_size_gpu1, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size + batch_size_gpu1, shuffle=False, num_workers=num_workers)

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = len(train_dataset.label_map)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for images, targets in progress_bar:
            # Split batch: half to GPU 0, double batch to GPU 1
            images0, images1 = images[:batch_size].to(device0), images[batch_size:].to(device1)
            targets0, targets1 = targets[:batch_size].to(device0), targets[batch_size:].to(device1)

            # Use separate streams for each GPU
            stream0 = torch.cuda.Stream(device0)
            stream1 = torch.cuda.Stream(device1)

            with torch.cuda.stream(stream0):
                outputs0 = model(images0)
                loss0 = criterion(outputs0, targets0)

            with torch.cuda.stream(stream1):
                outputs1 = model(images1)
                loss1 = criterion(outputs1, targets1)

            # Sync streams and combine losses
            torch.cuda.synchronize()
            loss = loss0 + loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), "resnet50_covid_trained.pth")
    print("Model weights saved to resnet50_covid_trained.pth")

    # Evaluation
    model.eval()
    all_targets = []
    all_outputs = []
    total = 0
    correct = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device0)
            targets = targets.to(device0)
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


if __name__ == "__main__":
    main()
