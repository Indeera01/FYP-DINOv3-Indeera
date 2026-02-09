import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

# -------------------------------
# Configuration
# -------------------------------
TEST_FOLDER = "embeddings-test"
MODEL_PATH = "temporal_cnn_dino_125.pth"
BATCH_SIZE = 8
NUM_FRAMES = 125
EMB_DIM = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("🧪 TESTING TEMPORAL CNN MODEL FOR DEEPFAKE DETECTION")
print("=" * 80)
print(f"📁 Test embeddings folder: {TEST_FOLDER}")
print(f"🤖 Model path: {MODEL_PATH}")
print(f"🖥️  Device: {DEVICE}")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"🎬 Expected frames per video: {NUM_FRAMES}, Embedding dim: {EMB_DIM}")
print("=" * 80)
print()

# -------------------------------
# Dataset
# -------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        self.paths = [os.path.join(folder, f) for f in self.files]
        self.labels = [1 if "generated" in f else 0 for f in self.files]
        
        print(f"📂 Found {len(self.files)} embedding files in test set")
        print(f"   Real videos: {self.labels.count(0)}")
        print(f"   Generated videos: {self.labels.count(1)}")
        print()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])  # Shape: (NUM_FRAMES, EMB_DIM)
        
        # Validate shape
        if x.shape != (NUM_FRAMES, EMB_DIM):
            raise ValueError(
                f"Expected shape ({NUM_FRAMES}, {EMB_DIM}), got {x.shape} in {self.files[idx]}"
            )
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, self.files[idx]


# -------------------------------
# Temporal CNN Model (same as training)
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
# Load Model
# -------------------------------
print("⏳ Loading trained model...")
model = TemporalCNN().to(DEVICE)

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found!")
    print("   Please train the model using train_temporal_cnn.py first.")
    exit(1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded successfully from {MODEL_PATH}")
print()

# -------------------------------
# Prepare test data
# -------------------------------
if not os.path.exists(TEST_FOLDER):
    print(f"❌ ERROR: Test folder '{TEST_FOLDER}' not found!")
    exit(1)

test_dataset = EmbeddingDataset(TEST_FOLDER)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 Test dataset summary:")
print(f"   Total samples: {len(test_dataset)}")
print(f"   Real videos: {test_dataset.labels.count(0)}")
print(f"   Generated videos: {test_dataset.labels.count(1)}")
print(f"   Batches: {len(test_loader)}")
print()

# -------------------------------
# Run Testing
# -------------------------------
print("🔬 Running inference on test set...")
print()

start_time = time.time()

all_preds = []
all_targets = []
all_filenames = []
all_probabilities = []
misclassified = []

with torch.no_grad():
    for batch_idx, (x, y, filenames) in enumerate(test_loader):
        x = x.to(DEVICE)
        
        # Get model outputs
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Store results
        all_preds.extend(predictions)
        all_targets.extend(y.numpy())
        all_filenames.extend(filenames)
        all_probabilities.extend(probabilities.cpu().numpy())
        
        # Track misclassifications
        for i, (pred, target, filename) in enumerate(zip(predictions, y.numpy(), filenames)):
            if pred != target:
                prob = probabilities[i].cpu().numpy()
                misclassified.append({
                    'filename': filename,
                    'true_label': 'Real' if target == 0 else 'Generated',
                    'predicted_label': 'Real' if pred == 0 else 'Generated',
                    'confidence': prob[pred] * 100
                })
        
        # Progress update
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
            print(f"   Processed {(batch_idx + 1) * BATCH_SIZE}/{len(test_dataset)} samples...", end='\r')

inference_time = time.time() - start_time
print()
print(f"✅ Inference completed in {inference_time:.2f} seconds")
print(f"   Average time per sample: {inference_time / len(test_dataset) * 1000:.2f} ms")
print()

# -------------------------------
# Calculate Metrics
# -------------------------------
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probabilities = np.array(all_probabilities)

# Overall metrics
overall_acc = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='binary')
recall = recall_score(all_targets, all_preds, average='binary')
f1 = f1_score(all_targets, all_preds, average='binary')

# Confusion matrix
cm = confusion_matrix(all_targets, all_preds)
tn, fp, fn, tp = cm.ravel()

# Per-class metrics
real_mask = all_targets == 0
generated_mask = all_targets == 1

real_correct = np.sum((all_preds[real_mask] == 0))
real_total = np.sum(real_mask)
real_acc = real_correct / real_total if real_total > 0 else 0

generated_correct = np.sum((all_preds[generated_mask] == 1))
generated_total = np.sum(generated_mask)
generated_acc = generated_correct / generated_total if generated_total > 0 else 0

# -------------------------------
# Display Results
# -------------------------------
print("=" * 80)
print("📊 COMPREHENSIVE TEST RESULTS")
print("=" * 80)
print()

print("🎯 OVERALL PERFORMANCE METRICS:")
print(f"   • Overall Accuracy:  {overall_acc * 100:.2f}%")
print(f"   • Precision:         {precision * 100:.2f}%")
print(f"   • Recall:            {recall * 100:.2f}%")
print(f"   • F1-Score:          {f1 * 100:.2f}%")
print()

print("📈 CONFUSION MATRIX:")
print(f"                    Predicted")
print(f"                Real    Generated")
print(f"   Actual Real    {tn:4d}      {fp:4d}")
print(f"         Gen      {fn:4d}      {tp:4d}")
print()

print("✅ CORRECTLY CLASSIFIED:")
print(f"   Total: {np.sum(all_preds == all_targets)}/{len(all_targets)} videos ({overall_acc * 100:.2f}%)")
print()

print("❌ MISCLASSIFIED:")
print(f"   Total: {np.sum(all_preds != all_targets)}/{len(all_targets)} videos ({(1 - overall_acc) * 100:.2f}%)")
print()

print("🎬 REAL VIDEOS PERFORMANCE (Class 0):")
print(f"   • Total real videos:                    {real_total}")
print(f"   • Correctly identified as real:         {real_correct} ({real_acc * 100:.2f}%)")
print(f"   • Misclassified as generated (FP):      {real_total - real_correct} ({(1 - real_acc) * 100:.2f}%)")
print(f"   • Specificity (True Negative Rate):     {tn / (tn + fp) * 100 if (tn + fp) > 0 else 0:.2f}%")
print()

print("🤖 GENERATED/DEEPFAKE VIDEOS PERFORMANCE (Class 1):")
print(f"   • Total generated videos:               {generated_total}")
print(f"   • Correctly identified as generated:    {generated_correct} ({generated_acc * 100:.2f}%)")
print(f"   • Misclassified as real (FN):           {generated_total - generated_correct} ({(1 - generated_acc) * 100:.2f}%)")
print(f"   • Sensitivity (True Positive Rate):     {tp / (tp + fn) * 100 if (tp + fn) > 0 else 0:.2f}%")
print()

# -------------------------------
# Detailed Classification Report
# -------------------------------
print("=" * 80)
print("📋 DETAILED CLASSIFICATION REPORT:")
print("=" * 80)
print()
print(classification_report(
    all_targets, 
    all_preds, 
    target_names=['Real', 'Generated'],
    digits=4
))

# -------------------------------
# Show misclassified samples
# -------------------------------
if misclassified:
    print("=" * 80)
    print(f"🔍 MISCLASSIFIED SAMPLES ({len(misclassified)} total):")
    print("=" * 80)
    print()
    
    # Show first 20 misclassifications
    num_to_show = min(20, len(misclassified))
    print(f"Showing first {num_to_show} misclassifications:\n")
    
    for i, item in enumerate(misclassified[:num_to_show], 1):
        print(f"{i:2d}. {item['filename']}")
        print(f"    True Label:      {item['true_label']}")
        print(f"    Predicted:       {item['predicted_label']}")
        print(f"    Confidence:      {item['confidence']:.2f}%")
        print()
    
    if len(misclassified) > num_to_show:
        print(f"... and {len(misclassified) - num_to_show} more misclassified samples")
        print()

# -------------------------------
# Confidence Statistics
# -------------------------------
print("=" * 80)
print("📊 PREDICTION CONFIDENCE STATISTICS:")
print("=" * 80)
print()

# Calculate confidence for correct and incorrect predictions
correct_mask = all_preds == all_targets
correct_confidences = np.max(all_probabilities[correct_mask], axis=1) * 100
incorrect_confidences = np.max(all_probabilities[~correct_mask], axis=1) * 100

print("✅ Correct Predictions:")
print(f"   • Mean confidence:    {np.mean(correct_confidences):.2f}%")
print(f"   • Median confidence:  {np.median(correct_confidences):.2f}%")
print(f"   • Min confidence:     {np.min(correct_confidences):.2f}%")
print(f"   • Max confidence:     {np.max(correct_confidences):.2f}%")
print()

if len(incorrect_confidences) > 0:
    print("❌ Incorrect Predictions:")
    print(f"   • Mean confidence:    {np.mean(incorrect_confidences):.2f}%")
    print(f"   • Median confidence:  {np.median(incorrect_confidences):.2f}%")
    print(f"   • Min confidence:     {np.min(incorrect_confidences):.2f}%")
    print(f"   • Max confidence:     {np.max(incorrect_confidences):.2f}%")
    print()

# -------------------------------
# Summary
# -------------------------------
print("=" * 80)
print("✨ TESTING COMPLETE!")
print("=" * 80)
print()
print(f"📌 Key Takeaways:")
print(f"   • Model achieved {overall_acc * 100:.2f}% accuracy on {len(test_dataset)} test videos")
print(f"   • Real video detection rate: {real_acc * 100:.2f}%")
print(f"   • Deepfake detection rate: {generated_acc * 100:.2f}%")
print(f"   • F1-Score: {f1 * 100:.2f}%")
print(f"   • Average inference time: {inference_time / len(test_dataset) * 1000:.2f} ms per video")
print()
print("=" * 80)
