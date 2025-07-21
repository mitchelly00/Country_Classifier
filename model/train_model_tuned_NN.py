import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_pickle("./data_splits/train.pkl")
df_val = pd.read_pickle("./data_splits/val.pkl")
df_test = pd.read_pickle("./data_splits/test.pkl")

label_col = "Dance"      # Target labels
feature_col = "Combined" # Your input feature vectors (list or np.array)

# Convert list of arrays into a 2D NumPy array
X_train = np.vstack(df_train[feature_col].values).astype(np.float32)
y_train = df_train[label_col].values

X_val = np.vstack(df_val[feature_col].values).astype(np.float32)
y_val = df_val[label_col].values

X_test = np.vstack(df_test[feature_col].values).astype(np.float32)
y_test = df_test[label_col].values


# Combine train and val for final training
X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)

# ----- Encode labels if not already -----
label_encoder = LabelEncoder()
y_combined = label_encoder.fit_transform(y_combined)
y_test_encoded = label_encoder.transform(y_test)

# ----- Convert to PyTorch tensors -----
X_train_tensor = torch.tensor(X_combined, dtype=torch.float32)
y_train_tensor = torch.tensor(y_combined, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----- Neural Network Model -----
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 224)
        self.dropout = nn.Dropout(0.1849)
        self.output = nn.Linear(224, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

input_dim = X_combined.shape[1]
output_dim = len(np.unique(y_combined))

model = NeuralNet(input_dim, output_dim)
device = torch.device("cpu")
model.to(device)

# ----- Training Setup -----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00315042)
n_epochs = 50  # Can adjust as needed

# ----- Training Loop -----
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss:.4f}")

# ----- Evaluation -----
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# ----- Results -----
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
