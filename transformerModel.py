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
from sklearn.metrics import roc_auc_score

# Function to get the GPUs with the most available memory using nvidia-smi
def get_best_gpus(num_gpus=4):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_memories = []
    for line in result.stdout.splitlines():
        index, memory = line.split(',')
        gpu_memories.append((int(index), int(memory.strip())))
    sorted_gpus = sorted(gpu_memories, key=lambda x: x[1], reverse=True)
    selected_gpus = [x[0] for x in sorted_gpus[:num_gpus]]
    return selected_gpus

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

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=32)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    num_classes = len(train_dataset.label_map)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

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

    torch.save(model.state_dict(), "vit_b16_trained.pth")
    print("Model weights saved to vit_b16_trained.pth")

    model.eval()
    all_targets = []
    all_outputs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.numel()
            correct += (predicted == targets).sum().item()
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    auc_score = roc_auc_score(all_targets, all_outputs, average='macro')

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print(f'Test AUC Score: {auc_score:.4f}')

if __name__ == "__main__":
    main()
