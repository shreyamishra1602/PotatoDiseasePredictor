#!/usr/bin/env python3
"""
Vision Transformer Training Script - MAC MPS OPTIMIZED
Trains ViT/DeiT model for potato leaf disease classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np


class PotatoDataset(Dataset):
    """Custom dataset for potato leaf images"""
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load class mapping
        with open("data/processed/metadata.json", "r") as f:
            metadata = json.load(f)
            self.class_to_idx = metadata["classes"]

        # Load all images
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_path / class_name
            if class_dir.exists():
                for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg"):
                    for img_path in class_dir.glob(ext):
                        self.images.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def get_device():
    """Get best available device - MPS for Mac, CUDA for NVIDIA, CPU fallback"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
        # MPS works best with float32
        torch.set_default_dtype(torch.float32)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_model(num_classes=3, model_name='vit_small_patch16_224'):
    """Create ViT model using timm"""
    print(f"\nCreating model: {model_name}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    val_loss = running_loss / total
    val_acc = 100. * correct / total

    return val_loss, val_acc


def main():
    print("=" * 60)
    print("ViT TRAINING - MAC MPS OPTIMIZED")
    print("=" * 60)

    # Configuration
    BATCH_SIZE = 32   # Good for most MacBooks; reduce to 16 if OOM
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    MODEL_NAME = 'vit_small_patch16_224'

    try:
        device = get_device()
        is_mps = device.type == "mps"

        # DataLoader settings for Mac
        # pin_memory is CUDA-only; num_workers=0 avoids macOS fork issues
        pin_memory = not is_mps and device.type == "cuda"
        num_workers = 0 if is_mps else 2

        # Load data
        print("\nLoading datasets...")
        train_transform, val_transform = get_transforms(IMAGE_SIZE)

        train_dataset = PotatoDataset("data/processed/train", transform=train_transform)
        val_dataset = PotatoDataset("data/processed/val", transform=val_transform)

        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")

        if len(train_dataset) == 0:
            print("No training images found! Run scripts/4_preprocess_images.py first.")
            return 1

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        # Create model
        model = create_model(num_classes=3, model_name=MODEL_NAME)
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Training loop
        print(f"\nStarting training for {EPOCHS} epochs on {device}...")
        best_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 60)

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = Path("data/models/vit_best.pth")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_name': MODEL_NAME,
                    'num_classes': 3
                }, model_path)
                print(f"   Best model saved! (Val Acc: {val_acc:.2f}%)")

        # Save final model
        final_path = Path("data/models/vit_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': MODEL_NAME,
            'num_classes': 3
        }, final_path)

        # Save training history
        import json as _json
        with open("data/models/training_history.json", "w") as f:
            _json.dump(history, f, indent=2)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"Models saved to: data/models/")
        print(f"Next: Run python scripts/6_test_models.py")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
