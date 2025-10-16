"""
EEG Motor Imagery Classification with Shallow ConvNet
-----------------------------------------------------
This script trains and evaluates a ShallowFBCSPNet model
on motor imagery EEG data stored in .mat files.
"""

# === 1. Imports ===
import scipy.io
import numpy as np
import mne
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from braindecode.models import ShallowFBCSPNet
import torch.nn as nn
import pandas as pd

# === 2. Data Loading and Epoching ===
# Load .mat EEG files, create Raw objects, extract events, and epoch the data
files = [
    "data/raw_mat/HaLTSubjectA1602236StLRHandLegTongue.mat",
    "data/raw_mat/HaLTSubjectA1603086StLRHandLegTongue.mat",

]

all_epochs = []
for f in files:
    mat = scipy.io.loadmat(f)
    content = mat['o'][0, 0]

    labels = content[4].flatten()
    signals = content[5]
    chan_names_raw = content[6]
    channels = [ch[0][0] for ch in chan_names_raw]
    fs = int(content[2][0, 0])

    df = pd.DataFrame(signals, columns=channels).drop(columns=["X5"], errors="ignore")
    eeg = df.values.T
    ch_names = df.columns.tolist()

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info)

    # Create events
    onsets = np.where((labels[1:] != 0) & (labels[:-1] == 0))[0] + 1
    event_codes = labels[onsets].astype(int)
    events = np.c_[onsets, np.zeros_like(onsets), event_codes]

    # Keep only relevant events
    mask = np.isin(events[:, 2], np.arange(1, 7))
    events = events[mask]

    event_id = {
        "left_hand": 1,
        "right_hand": 2,
        "neutral": 3,
        "left_leg": 4,
        "tongue": 5,
        "right_leg": 6,
    }

    # Epoching
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0,
        tmax=1.5,
        baseline=None,
        preload=True,
    )

    all_epochs.append(epochs)

epochs_all = mne.concatenate_epochs(all_epochs)

# === 3. Minimal Preprocessing + Train/Validation Split ===
# Convert epochs to numpy arrays (N, C, T) and split into train/val sets
X = epochs_all.get_data().astype("float32")
y = (epochs_all.events[:, -1] - 1).astype("int64")  # classes 0..5

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Torch DataLoaders ===
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# === 5. Model – Shallow ConvNet ===
# Reference: Schirrmeister et al. (2017)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ShallowFBCSPNet(
    n_chans=X.shape[1],
    n_outputs=len(np.unique(y)),
    n_times=X.shape[2],
    final_conv_length="auto"
).to(device)

# Load pretrained weights
state_dict = torch.load("shallow_weights_all.pth", map_location=device)
model.load_state_dict(state_dict)

# === 6. Training ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 21):
    # Training
    model.train()
    correct, total = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch:02d} | Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")
    
    #save model
    torch.save(model, "model.pth")
    print("✅ Model saved as model.pth")
