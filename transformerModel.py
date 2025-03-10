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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Load pretrained Vision Transformer
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Adjust the classifier head for multi-label classification
    num_classes = len(train_dataset.label_map)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Wrap model with DataParallel for multi-GPU usage
    model = nn.DataParallel(model, device_ids=[4, 5, 6])  # GPUs 4, 5, 6

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

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()
