import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------
# MLP Neural Network
# ---------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden_dim, num_layers, dropout=0.2, batchnorm=False):
        super().__init__()
        layers = []

        layers.append(nn.Linear(d_in, hidden_dim))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))   # use batchnorm to speed up training and improve stability
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))      #using the dropout technique to avoid overfitting

        for _ in range(num_layers - 2):     # num_layers - 2 because we have the input and output layer as well
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))   # use batchnorm to speed up training and improve stability
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  #using the dropout technique to avoid overfitting

        layers.append(nn.Linear(hidden_dim, n_out))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(X, y, batch_size=32):
    # convert X to a PyTorch tensor of type float32
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # convert y to a PyTorch tensor of type long
    y_tensor = torch.tensor(y, dtype=torch.long)

    # create a dataset to pair each feature with the corresponding label
    dataset = TensorDataset(X_tensor, y_tensor)

    # wrap the dataset in a dataLoader for batching and shuffling during training
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------
# Training function
# ---------------------------------------------------
def train_model(model, loader, epochs=10, lr=1e-3):
    device = get_device()
    model = model.to(device) # move the model to the selected device

    model.train() 
    
    opt = torch.optim.Adam(model.parameters(), lr=lr) # create an optimizer (Adam)
    loss_fn = nn.CrossEntropyLoss() # define the loss function

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # move data to the same device as model

            opt.zero_grad()        # reset gradients before each batch
            logits = model(x)      # compute predictions
            loss = loss_fn(logits, y)  # compute loss

            loss.backward()        # compute gradients
            opt.step()             # update model parameters

            total_loss += loss.item()  # add loss

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}")

    print("[OK] Training complete.")
    return model