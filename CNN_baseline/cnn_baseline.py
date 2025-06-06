import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

class CNNClassifier(nn.Module):
    """
    A small CNN for classifying 40x40 grayscale images
    """
    def __init__(self, num_classes=10):  # Change `num_classes` as needed
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: 16x40x40
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the dimensions
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)       # Output: 32x20x20
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)       # Output: 64x10x10

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16x20x20
        x = self.pool(F.relu(self.conv2(x)))  # -> 32x10x10
        x = self.pool(F.relu(self.conv3(x)))  # -> 64x5x5
        x = x.view(-1, 64 * 5 * 5)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, len(train_dataset.classes)

def build_model(num_classes, model_name='resnet'):
    if model_name == 'resnet':
        print("Buildeing Resnet18 model...")
        model = resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
    else:
        print("Building custom small CNN model...")
        model = CNNClassifier(num_classes)
    return model

def train_model(model, train_loader, test_loader, device, epochs=100, patience=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # ===== Test Evaluation =====
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100. * test_correct / test_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

        # ===== Early Stopping =====
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_resnet_model.pt")
            print("New best model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on 40x40 grayscale images")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--image_size", type=int, default=40)
    parser.add_argument("--model_name", type=str, default='resnet')
    args = parser.parse_args()

    start = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_loader, test_loader, num_classes = get_data_loaders(args.data_dir, args.image_size, args.batch_size)
    print(f"Dataset {args.data_dir} has {num_classes} classes")
    model = build_model(num_classes, model_name=args.model_name).to(device)

    train_model(model, train_loader, test_loader, device,
                epochs=args.epochs, patience=args.patience, lr=args.lr)

    delta = datetime.now() - start
    print("Time elapsed: ", delta)

if __name__ == "__main__":
    main()
