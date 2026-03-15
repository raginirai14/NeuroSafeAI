import os, argparse, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_transforms(mode="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def get_dataloaders(data_dir, batch_size=16):
    dataloaders = {}
    dataset_sizes = {}
    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        dataset = datasets.ImageFolder(root=split_dir, transform=get_transforms(split))
        dataset_sizes[split] = len(dataset)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=0)
    print(f"Classes: {dataloaders['train'].dataset.classes}")
    print(f"Sizes:   {dataset_sizes}")
    return dataloaders, dataset_sizes

def get_model():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return backbone

def train(data_dir, epochs, batch_size, lr, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    dataloaders, dataset_sizes = get_dataloaders(data_dir, batch_size)
    model = get_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} " + "-"*30)
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_correct = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (outputs > 0.5).float()
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print(f"  {phase.upper():5}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
                print(f"  >>> Best model saved! Acc: {best_acc:.4f}")
        scheduler.step()
    print(f"\nDone! Best Val Accuracy: {best_acc:.4f}")
    print(f"Model saved at: {save_dir}/best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../datasets")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir)