import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
from glob import glob

class ChestXrayDataset(Dataset):
    def __init__(self, images_folder, csv_file, transform=None, subset_list=None):
        self.images_folder = images_folder
        self.transform = transform
        self.data = self.load_data(csv_file, subset_list)
        self.label_map = self.create_label_map()

    def load_data(self, csv_file, subset_list):
        # Load main CSV file
        df = pd.read_csv(csv_file)
        print("Sample Image Index from CSV:", df['Image Index'].head())
        
        # Filter based on subset list (train or test)
        if subset_list:
            with open(subset_list, 'r') as f:
                image_list = [line.strip() for line in f.readlines()]
            df = df[df['Image Index'].isin(image_list)]

        # Map image file names to paths and filter rows with missing images
        df['path'] = df['Image Index'].map(self.get_image_paths())
        df.dropna(subset=['path'], inplace=True)  # Remove entries without valid image paths

        print(f"Loaded dataset with {len(df)} images after filtering")
        return df

    def get_image_paths(self):
        # Set the search path based on the corrected structure
        search_path = os.path.join(self.images_folder, 'images_*/images', '*.png')
        
        # Collect image paths
        image_paths = {os.path.basename(x): x for x in glob(search_path)}

        print(f"Found {len(image_paths)} images.")
        if len(image_paths) > 0:
            print("Sample image paths:", list(image_paths.items())[:5])
        
        return image_paths

    def create_label_map(self):
        # Collect unique labels for multi-label classification
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

        # Load and process image
        image = Image.open(img_path).convert('RGB')
        labels = row['Finding Labels'].split('|')
        target = torch.zeros(len(self.label_map))
        
        # Encode labels as multi-hot vector
        for label in labels:
            if label in self.label_map:
                target[self.label_map[label]] = 1

        if self.transform:
            image = self.transform(image)

        return image, target

def main():
    # Paths to data directories and files
    data_folder = "/shared/storage/cs/studentscratch/ay841/nih-chest-xrays"
    csv_file = os.path.join(data_folder, "Data_Entry_2017.csv")
    train_list = os.path.join(data_folder, "train_val_list.txt")
    test_list = os.path.join(data_folder, "test_list.txt")

    # Define data transformations
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

    # Create datasets
    train_dataset = ChestXrayDataset(images_folder=data_folder,
                                     csv_file=csv_file,
                                     transform=transform_train,
                                     subset_list=train_list)
    test_dataset = ChestXrayDataset(images_folder=data_folder,
                                    csv_file=csv_file,
                                    transform=transform_test,
                                    subset_list=test_list)

    # Print the length of the datasets
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load pretrained ResNet-18 model
    model = models.resnet18()  # Create model without weights
    num_classes = len(train_dataset.label_map)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust final layer for multi-label output

    # Load manually downloaded weights
    weights_path = "/shared/storage/cs/studentscratch/ay841/torch_cache/hub/checkpoints/resnet18-f37072fd.pth"
    model.load_state_dict(torch.load(weights_path))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()
