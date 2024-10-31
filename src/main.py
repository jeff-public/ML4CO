import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from GraphDataset import GraphDataset
from InterleavedGCNN import InterleavedGCNN
from parentBCELoss import weightedBCEWithLogitsLoss

from train_val_test import train, validate, test



if __name__ == '__main__':

    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set hyper-parameters
    hidden_channels = 128
    num_layers = 3

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Training data
    # Load children data
    batch_size = 50
    children_dir = f'./../dataset/training/children_graphs'
    children_data = GraphDataset(root=children_dir, data_list=[])
    children_loader = DataLoader(children_data, batch_size=batch_size, shuffle=True)

    # Load parents dataset
    batch_size = 10
    parents_dir = f'./../dataset/training/parents_graphs'
    parents_data = GraphDataset(root=parents_dir, data_list=[])
    parents_loader = DataLoader(parents_data, batch_size=batch_size, shuffle=True)


    ## Validation data
    val_dir = f"./../dataset/validation"
    val_data = GraphDataset(root=val_dir, data_list=[])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


    ## Testing data
    test_dir = f"./../dataset/testing"
    test_data = GraphDataset(root = test_dir, data_list = [])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    


    model = InterleavedGCNN(
        var_in_channels=children_loader.dataset[0]["var_nodes"].x.shape[1],
        cons_in_channels=children_loader.dataset[0]["constr_nodes"].x.shape[1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        edge_attr_dim=1
    ).to(device)

    # Define loss function and optimizer
    criterion = {"children": nn.BCEWithLogitsLoss(pos_weight = 0.1),
                 "parents": weightedBCEWithLogitsLoss()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    ## Training and validation
    num_epochs = 10
    train_loss = []
    val_loss = []
    for epoch in range(1, num_epochs + 1):
        losses = train(model = model, children_loader = children_loader, 
                       parents_loader = parents_loader, 
                       children_criterion = criterion["children"],
                       parents_criterion = criterion["parents"],
                       optimizer = optimizer, device = device)
        train_loss.append(losses[0])
        val_loss.append(losses[1])
        print(f"Epoch: {epoch:02d}, Train Loss: {losses[0]:.4f}, Val Loss: {losses[1]:.4f}")









    # ## Testing 
    # with torch.no_grad():
    #     for data in test_loader:
    #         data = data.to(device)
    #         out = model(data)
    #         loss = criterion(out[data["var_nodes"].mask],
    #                          data['var_nodes'].y[data["var_nodes"].mask])
    #         total_loss += loss.item() * data['var_nodes'].num_nodes

        




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