import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# -------------------------------
# Configuration
# -------------------------------
TRAIN_FOLDER = "embeddings-train"
VAL_FOLDER = "embeddings-validation"
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
NUM_FRAMES = 125
EMB_DIM = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("🚀 Temporal CNN Training for Deepfake Detection")
print("=" * 60)
print(f"📁 Training folder: {TRAIN_FOLDER}")
print(f"📁 Validation folder: {VAL_FOLDER}")
print(f"🖥️  Device: {DEVICE}")
print(f"📊 Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, Learning rate: {LR}")
print(f"🎬 Expected frames per video: {NUM_FRAMES}, Embedding dim: {EMB_DIM}")
print("=" * 60)

# -------------------------------
# Dataset
# -------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        self.paths = [os.path.join(folder, f) for f in self.files]
        self.labels = [1 if "generated" in f else 0 for f in self.files]
        
        print(f"📂 Found {len(self.files)} embedding files")
        print(f"   Real videos: {self.labels.count(0)}")
        print(f"   Generated videos: {self.labels.count(1)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])  # Shape: (NUM_FRAMES, EMB_DIM)
        
        # Validate shape (preprocess_videos.py ensures exactly NUM_FRAMES frames)
        if x.shape != (NUM_FRAMES, EMB_DIM):
            raise ValueError(
                f"Expected shape ({NUM_FRAMES}, {EMB_DIM}), got {x.shape} in {self.files[idx]}"
            )
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# -------------------------------
# Temporal CNN Model
# -------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, num_classes=2):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv1d(emb_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# -------------------------------
# Prepare data
# -------------------------------
train_dataset = EmbeddingDataset(TRAIN_FOLDER)
val_dataset = EmbeddingDataset(VAL_FOLDER)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n📊 Dataset summary:")
print(f"   Training samples: {len(train_dataset)} ({train_dataset.labels.count(0)} real, {train_dataset.labels.count(1)} generated)")
print(f"   Validation samples: {len(val_dataset)} ({val_dataset.labels.count(0)} real, {val_dataset.labels.count(1)} generated)")
print(f"   Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
print()

# -------------------------------
# Training setup
# -------------------------------
model = TemporalCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# Training loop
# -------------------------------
print("🏋️  Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ---------------------------
    # Validation
    # ---------------------------
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            outputs = model(x)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(pred)
            targets.extend(y.numpy())

    acc = accuracy_score(targets, preds)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

# -------------------------------
# Final Validation Results
# -------------------------------
print("\n" + "=" * 60)
print("📊 FINAL VALIDATION RESULTS")
print("=" * 60)

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(pred)
        all_targets.extend(y.numpy())

# Calculate detailed metrics
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

overall_acc = accuracy_score(all_targets, all_preds)

# Per-class accuracy
real_mask = all_targets == 0
generated_mask = all_targets == 1

real_correct = np.sum((all_preds[real_mask] == 0))
real_total = np.sum(real_mask)
real_acc = real_correct / real_total if real_total > 0 else 0

generated_correct = np.sum((all_preds[generated_mask] == 1))
generated_total = np.sum(generated_mask)
generated_acc = generated_correct / generated_total if generated_total > 0 else 0

print(f"Overall Validation Accuracy: {overall_acc * 100:.2f}%")
print(f"  ✓ Correctly classified: {np.sum(all_preds == all_targets)}/{len(all_targets)} videos")
print(f"  ✗ Misclassified: {np.sum(all_preds != all_targets)}/{len(all_targets)} videos")
print()
print(f"Real Videos (Class 0):")
print(f"  ✓ Correctly identified as real: {real_correct}/{real_total} ({real_acc * 100:.2f}%)")
print(f"  ✗ Misclassified as generated: {real_total - real_correct}/{real_total}")
print()
print(f"Generated/Deepfake Videos (Class 1):")
print(f"  ✓ Correctly identified as generated: {generated_correct}/{generated_total} ({generated_acc * 100:.2f}%)")
print(f"  ✗ Misclassified as real: {generated_total - generated_correct}/{generated_total}")
print("=" * 60)

# -------------------------------
# Save model
# -------------------------------
torch.save(model.state_dict(), "temporal_cnn_dino_125.pth")
print("\n✅ Model saved as temporal_cnn_dino_125.pth")
