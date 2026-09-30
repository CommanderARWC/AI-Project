import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# Streaming HDF5 Dataset (Prevents RAM Crash)
# =====================================================

class ModulationH5Dataset(Dataset):
    def __init__(self, file_path, indices):
        self.file_path = file_path
        self.indices = indices
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Open file lazily in the worker process
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, "r")
            
        real_idx = self.indices[idx]
        
        # Stream specific example from the disk
        x = self.h5_file["X"][real_idx]
        y = self.h5_file["Y"][real_idx]
        
        # ASSUMPTION: The HDF5 file contains a dataset "Z" for SNR values. 
        # Update "Z" to your specific key (e.g., "SNR") if it differs.
        snr = self.h5_file["SNR"][real_idx] 
        
        # Instance Normalization (Unit average power)
        power = np.mean(x ** 2)
        x = x / np.sqrt(power + 1e-12)
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.long), 
                torch.tensor(snr, dtype=torch.float32))

# =====================================================
# Data Splits
# =====================================================

FILE_PATH = "modulation_dataset2.h5"

with h5py.File(FILE_PATH, "r") as f:
    total_samples = f["X"].shape[0]

print(f"Total Dataset Size: {total_samples}")

np.random.seed(42)
all_indices = np.random.permutation(total_samples)

train_end = int(0.70 * total_samples)
val_end = int(0.85 * total_samples)

train_indices = all_indices[:train_end]
val_indices = all_indices[train_end:val_end]
test_indices = all_indices[val_end:]

print(f"Train: {len(train_indices)}  Val: {len(val_indices)}  Test: {len(test_indices)}")

train_loader = DataLoader(ModulationH5Dataset(FILE_PATH, train_indices), batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(ModulationH5Dataset(FILE_PATH, val_indices), batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(ModulationH5Dataset(FILE_PATH, test_indices), batch_size=256, shuffle=False, num_workers=0)

# =====================================================
# CNN + BiLSTM + Transformer Architecture
# =====================================================

class AMC_HybridNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 1) 
        x, _ = self.lstm(x) 
        x = self.transformer(x)
        x = x.permute(0, 2, 1) 
        pooled = self.pool(x)         
        pooled = torch.flatten(pooled, 1) 
        out = self.classifier(pooled)
        return out

model = AMC_HybridNet(num_classes=9).to(device)

# =====================================================
# Loss Function & Optimizer
# =====================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================
# Training 
# =====================================================

epochs = 15

train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_state = None

for epoch in range(epochs):
    model.train()
    running_loss = 0

    # Note the ignored _ for SNR during training
    for batch_X, batch_Y, _ in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_Y, _ in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_Y)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == batch_Y).sum().item()
            val_total += batch_Y.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss = {avg_train_loss:.4f}  Val Loss = {avg_val_loss:.4f}  Val Acc = {val_acc:.2f}%")

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nRestored best model (Val Acc = {best_val_acc:.2f}%)")

# =====================================================
# Testing & Collecting SNR Data
# =====================================================

model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []
all_snrs = []

with torch.no_grad():
    for batch_X, batch_Y, batch_SNR in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == batch_Y).sum().item()
        total += batch_Y.size(0)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_Y.cpu().numpy())
        
        # Flatten SNR in case it has an extra dimension
        all_snrs.extend(batch_SNR.view(-1).numpy()) 

accuracy = 100 * correct / total
print(f"\nOverall Test Accuracy: {accuracy:.2f}%")

# =====================================================
# Plot 1: Train vs Validation Loss (Epochs)
# =====================================================

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss vs. Epoch')
plt.legend()
plt.tight_layout()
plt.savefig("loss_vs_epoch.png")
plt.show()

# =====================================================
# Plot 2: Confusion Matrix
# =====================================================

cm = confusion_matrix(all_labels, all_preds)
mod_names = ["OOK", "ASK", "BPSK", "QPSK", "8PSK", "16PSK", "BFSK", "4FSK", "8FSK"]

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="viridis")
plt.colorbar()
plt.xticks(np.arange(len(mod_names)), mod_names, rotation=45)
plt.yticks(np.arange(len(mod_names)), mod_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (CNN + BiLSTM + Transformer)")

for i in range(len(mod_names)):
    for j in range(len(mod_names)):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] < (cm.max()/2) else "black")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# =====================================================
# Plot 3: Accuracy vs SNR
# =====================================================

snr_accuracy = defaultdict(list)

# Group correctness by SNR
for snr, true_label, pred_label in zip(all_snrs, all_labels, all_preds):
    is_correct = (true_label == pred_label)
    snr_accuracy[snr].append(is_correct)

# Calculate accuracy percentage per SNR
unique_snrs = sorted(snr_accuracy.keys())
acc_per_snr = []

for snr in unique_snrs:
    correct_count = sum(snr_accuracy[snr])
    total_count = len(snr_accuracy[snr])
    acc_per_snr.append(100.0 * correct_count / total_count)

plt.figure(figsize=(8, 6))
plt.plot(unique_snrs, acc_per_snr, marker='o', linestyle='-', color='b')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.title('Classification Accuracy vs. SNR')
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_snr.png")
plt.show()

torch.save(model, "model_weights.pt")
print("Model weights saved successfully as model_weights.pt!")
