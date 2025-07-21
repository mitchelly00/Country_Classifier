import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import optuna

# Load data
train_df = pd.read_pickle("./data_splits/train.pkl")
val_df = pd.read_pickle("./data_splits/val.pkl")
test_df = pd.read_pickle("./data_splits/test.pkl")  # Keep for later final evaluation

# Assuming features are in a column "Combined" as np arrays, labels in 'Dance'
def prepare_dataset(df):
    X = np.vstack(df["Combined"].values)
    y = df["Dance"].values
    return X, y

X_train, y_train = prepare_dataset(train_df)
X_val, y_val = prepare_dataset(val_df)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val_enc, dtype=torch.long)

# Create datasets and loaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

# Device is CPU explicitly
device = torch.device("cpu")

# Define a simple feed-forward NN with hyperparameters from Optuna
class Net(nn.Module):
    def __init__(self, input_dim, n_layers, n_units, dropout):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = n_units
        layers.append(nn.Linear(current_dim, len(le.classes_)))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def objective(trial):
    # Hyperparameters to tune
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = Net(X_train.shape[1], n_layers, n_units, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    epochs = 50

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
                val_steps += 1
        val_loss /= val_steps

        trial.report(val_loss, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
