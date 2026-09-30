import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# Load Dataset
# =====================================================

with h5py.File("modulation_dataset2.h5", "r") as f:
    X = f["X"][:]
    Y = f["Y"][:]
    SNR = f["SNR"][:]

X = torch.tensor(X, dtype=torch.float32)   # shape: (N, 2, 1024)  -> I/Q already as 2 "rows"
Y = torch.tensor(Y, dtype=torch.long)
SNR = torch.tensor(SNR, dtype=torch.int32)

print("X Shape:", X.shape)
print("Y Shape:", Y.shape)
print("SNR Shape:", SNR.shape)

# =====================================================
# FIX: Input Normalization
# =====================================================
# Each example is normalized to unit average power. Without this, the CNN
# partially learns to use raw amplitude as an SNR/power proxy rather than
# focusing purely on modulation-discriminative shape, and training can be
# less stable across the wide SNR range (-20 to 20 dB) in this dataset.

power = torch.mean(X ** 2, dim=(1, 2), keepdim=True)  # avg power per example
X = X / torch.sqrt(power + 1e-12)

# =====================================================
# Shuffle Dataset
# =====================================================

indices = torch.randperm(len(X))

X = X[indices]
Y = Y[indices]
SNR = SNR[indices]

# =====================================================
# FIX: Train / Validation / Test Split (was train/test only)
# =====================================================
# A validation set lets you monitor overfitting DURING training instead of
# only discovering it at the very end on the test set.

n = len(X)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]
SNR_test = SNR[val_end:]

print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# =====================================================
# Data Loaders
# =====================================================
# NOTE: no more .unsqueeze(1) -- X is already (N, 2, 1024), and we now feed
# that directly into Conv1d as (batch, in_channels=2, length=1024), with I
# and Q as the two channels instead of collapsing them via a Conv2d kernel.

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# =====================================================
# FIX: CNN Model -- Conv1d with I/Q as channels
# =====================================================
# OLD: Conv2d with kernel (2, 8) on input (1, 2, 1024) collapses the I/Q
# dimension entirely after the FIRST layer -- from then on the network has
# no separate "I" and "Q" representation left, just a flattened 1D feature
# map. That throws away structure a network could otherwise exploit
# (e.g. I/Q phase relationships) across depth.
#
# NEW: Conv1d treats I and Q as 2 input channels that persist and mix
# through the conv stack the way most published RadioML CNN architectures
# (e.g. O'Shea et al.) actually do it, and adds BatchNorm for more stable
# training given the -20..20 dB SNR range in this dataset.

class AMC_CNN(nn.Module):
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

        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AMC_CNN(num_classes=9).to(device)

# =====================================================
# Loss Function & Optimizer
# =====================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================
# Training (now tracks train + val loss/accuracy per epoch)
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

    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- Validation pass ----
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

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

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss = {avg_train_loss:.4f}  "
        f"Val Loss = {avg_val_loss:.4f}  "
        f"Val Acc = {val_acc:.2f}%"
    )

# Restore best-validation-accuracy weights before final testing
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nRestored best model (Val Acc = {best_val_acc:.2f}%)")

# =====================================================
# Plot Train vs Val Loss (new diagnostic plot)
# =====================================================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Testing
# =====================================================

model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_Y in test_loader:

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        outputs = model(batch_X)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == batch_Y).sum().item()

        total += batch_Y.size(0)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_Y.cpu().numpy())

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(all_labels, all_preds)

mod_names = [
    "OOK", "ASK", "BPSK", "QPSK", "8PSK", "16PSK", "BFSK", "4FSK", "8FSK"
]

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.colorbar()
plt.xticks(np.arange(len(mod_names)), mod_names, rotation=45)
plt.yticks(np.arange(len(mod_names)), mod_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

for i in range(len(mod_names)):
    for j in range(len(mod_names)):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

# =====================================================
# Accuracy vs SNR
# =====================================================

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

snr_test_np = SNR_test.numpy()
snr_values = np.unique(snr_test_np)

snr_accuracy = []

for snr in snr_values:
    mask = (snr_test_np == snr)
    correct = np.sum(all_preds[mask] == all_labels[mask])
    total = np.sum(mask)
    snr_accuracy.append(100 * correct / total)

plt.figure(figsize=(8, 5))
plt.plot(snr_values, snr_accuracy, marker="o")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy vs SNR")
plt.grid(True)
plt.show()

# =====================================================
# Normalized Confusion Matrix
# =====================================================

cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")

plt.figure(figsize=(10, 8))
plt.imshow(cm_norm, interpolation="nearest")
plt.colorbar()
plt.xticks(np.arange(len(mod_names)), mod_names, rotation=45)
plt.yticks(np.arange(len(mod_names)), mod_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")

for i in range(len(mod_names)):
    for j in range(len(mod_names)):
        plt.text(
            j, i, f"{cm_norm[i, j]:.2f}",
            ha="center", va="center",
            color="white" if cm_norm[i, j] > 0.5 else "black"
        )

plt.tight_layout()
plt.show()
torch.save(model,"model_weights.pt")
print("Model weights saved successfully as model_weights.pt!")
