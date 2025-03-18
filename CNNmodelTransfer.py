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
        label_map = {label: idx for idx, label in enumerate(labels)}
        return label_map

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
    model = models.resnet50(weights=None)  # No ImageNet weights since we're loading custom ones
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        # Load model safely with weights_only=True to avoid security issues
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Remove 'module.' prefix if model was saved with DataParallel
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key.replace('module.', '')
            if key not in ['fc.weight', 'fc.bias']:  # Skip fc layer since it's being redefined
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully!")
    else:
        print(f"Model file {model_path} not found. Starting from scratch.")
    
    return model

def main():
    data_folder = "/shared/storage/cs/studentscratch/ay841/nih-chest-xrays"
    csv_file = os.path.join(data_folder, "Data_Entry_2017.csv")
    train_list = os.path.join(data_folder, "train_val_list.txt")
    test_list = os.path.join(data_folder, "test_list.txt")
    model_path = "resnet50_covid_trained.pth"

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    train_dataset = ChestXrayDataset(images_folder=data_folder,
                                     csv_file=csv_file,
                                     transform=transform_train,
                                     subset_list=train_list)
    test_dataset = ChestXrayDataset(images_folder=data_folder,
                                    csv_file=csv_file,
                                    transform=transform_test,
                                    subset_list=test_list)

    batch_size = int(input("Enter batch size: "))
    num_workers = int(input("Enter number of workers: "))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.label_map)

    model = load_model(model_path, num_classes, device)

    # Freeze backbone first and train only the classifier
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 3  # Classifier training phase
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Unfreeze backbone and fine-tune whole model
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Fine-Tuning Epoch {epoch+1}/{num_epochs}")

        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f'Fine-Tuning Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), "resnet50_trained_nih.pth")
    print("Model weights saved to resnet50_trained_nih.pth")

if __name__ == "__main__":
    main()
