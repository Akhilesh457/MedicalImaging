"""
OPTIMIZED Fast Training Script for RTX 4050
All performance optimizations applied
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class IDCDataset(Dataset):
    """Dataset for Invasive Ductal Carcinoma (IDC) images"""
    
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.samples = []
        self.transform = transform

        for patient in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            for label in ["0", "1"]:
                label_path = os.path.join(patient_path, label)
                if not os.path.exists(label_path):
                    continue

                for img_name in os.listdir(label_path):
                    self.samples.append(
                        (os.path.join(label_path, img_name), int(label))
                    )
        
        # Limit dataset size for faster testing
        if max_samples:
            self.samples = self.samples[:max_samples]
            print(f"Limited to {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ExplainableViT(nn.Module):
    """Vision Transformer with explainability features"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ExplainableViT, self).__init__()
        
        if pretrained:
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = models.vit_b_16(weights=None)
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Replace classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []
        output = self.vit(x)
        return output


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cuda'):
    """Train the model with validation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Mixed precision training for speed
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, 'best_vit_idc_fast.pth')
            print(f"✓ Best model saved with Val Acc: {val_acc:.4f}")
    
    return history


def main():
    """Main training pipeline - OPTIMIZED"""
    
    # Configuration - OPTIMIZED FOR SPEED
    DATASET_PATH = "archive\IDC_regular_ps50_idx5"  # Update this
    BATCH_SIZE = 64  # Increased from 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 2  # REDUCED for Windows stability (was 4)
    
    # FOR TESTING: Use subset of data (remove max_samples=None for full dataset)
    USE_SUBSET = False  # Set to True for quick testing
    MAX_SAMPLES = 30000 if USE_SUBSET else None  # ~10% of data
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # SIMPLIFIED transforms - removed slow operations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Only keep simple flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = IDCDataset(DATASET_PATH, transform=None, max_samples=MAX_SAMPLES)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # OPTIMIZED dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS,  # CRITICAL
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Disabled for Windows stability
        persistent_workers=False  # Disabled for validation
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = ExplainableViT(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=device
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Time per epoch: {total_time/EPOCHS/60:.1f} minutes")
    print("="*70)


if __name__ == "__main__":
    # Windows multiprocessing fix
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()