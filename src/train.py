import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from InterleavedGCNN import InterleavedGCNN
from GraphDataset import GraphDataset

# Training and validation functions
def train(model, loader, optimizer, criterion, device):
    ## Phase I training
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data["var_nodes"].mask],
                         data['var_nodes'].y[data["var_nodes"].mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data['var_nodes'].num_nodes

    ## Phase II training



    return total_loss / len(loader.dataset)




def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out[data["var_nodes"].mask],
                             data['var_nodes'].y[data["var_nodes"].mask])
            total_loss += loss.item() * data['var_nodes'].num_nodes
    return total_loss / len(loader.dataset)



if __name__ == '__main__':

    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set hyper-parameters
    batch_size = 50
    hidden_channels = 128
    num_layers = 3

    # Load dataset
    train_dir = f'./../dataset/training/children_graphs'
    train_data = GraphDataset(root=train_dir, data_list=[])
    train_size = int(len(train_data) * 0.9)
    train_loader = DataLoader(train_data[:train_size], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_data[train_size:], batch_size=batch_size, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InterleavedGCNN(
        var_in_channels=train_loader.dataset[0]["var_nodes"].x.shape[1],
        cons_in_channels=train_loader.dataset[0]["constr_nodes"].x.shape[1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        edge_attr_dim=1
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_loss = []
    val_loss = []
    for epoch in range(1, num_epochs + 1):
        train_loss.append(train(model, train_loader, optimizer, criterion, device))
        val_loss.append(validate(model, val_loader, criterion, device))
        print(f"Epoch: {epoch:02d}, "
              f"Train Loss: {train_loss[epoch-1]:.4f}, "
               f"Val Loss: {val_loss[epoch-1]:.4f}")
        
    # Save the model
    torch.save(model, "./../models/InterleavedGCNN.pt")
        

    # Save loss plot
    plt.figure(figsize=(4, 3))
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epcohs")
    plt.ylabel("BCE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    plt.savefig("./loss_plot.png")
    plt.show()
    plt.close("all")