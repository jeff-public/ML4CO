import torch

# Training function
# Training function
def train(model, children_loader, 
          parents_loader, children_criterion, 
          parents_criterion, optimizer, device):
    """
    Function to train the model sequentially: first on children ILPs, then on parent ILPs,
    using different loss functions for each phase.
    
    Args:
    - model: PyTorch model to train.
    - children_loader: DataLoader for children ILPs.
    - parents_loader: DataLoader for parent ILPs.
    - children_criterion: Loss function for children ILPs.
    - parents_criterion: Loss function for parent ILPs.
    - optimizer: Optimizer for training.
    - device: Device to use for training (CPU/GPU).
    
    Returns:
    - avg_children_loss: Average loss for children ILPs.
    - avg_parents_loss: Average loss for parent ILPs.
    """
    # Phase I: Train on children ILPs with children-specific loss function
    avg_children_loss = None
    if children_loader:
        avg_children_loss = train_on_children_data(
            model, children_loader, optimizer, children_criterion, device)
    
    # Phase II: Train on parent ILPs with parent-specific loss function
    avg_parents_loss = None
    if parents_loader:
        avg_parents_loss = train_on_parents_data(
            model, parents_loader, optimizer, parents_criterion, device)
    
    return avg_children_loss, avg_parents_loss


def train_on_children_data(model, loader, optimizer, criterion, device):
    """
    Generic training loop for a given dataset (children or parent ILPs),
    using a specified loss function.
    
    Args:
    - model: PyTorch model to train.
    - loader: DataLoader for ILPs (children or parents).
    - optimizer: Optimizer for training.
    - criterion: Loss function for the specific phase.
    - device: Device to use for training (CPU/GPU).
    - phase: 'children' or 'parents', indicating the training phase.
    
    Returns:
    - avg_loss: Average loss over the dataset.
    """
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss

    for data in loader:
        data = data.to(device)  # Move data to the specified device
        optimizer.zero_grad()   # Reset gradients

        # Forward pass and compute loss
        out = model(data)
        loss = criterion(out[data["var_nodes"].mask],
                         data['var_nodes'].y[data["var_nodes"].mask])

        loss.backward()   # Backpropagation
        optimizer.step()  # Update model weights

        total_loss += loss.item() * data['var_nodes'].num_nodes  # Accumulate loss

    avg_loss = total_loss / len(loader.dataset)  # Calculate average loss

    return avg_loss


def train_on_parents_data(model, loader, optimizer, criterion, device):
    """
    Generic training loop for a given dataset (children or parent ILPs),
    using a specified loss function.
    
    Args:
    - model: PyTorch model to train.
    - loader: DataLoader for ILPs (children or parents).
    - optimizer: Optimizer for training.
    - criterion: Loss function for the specific phase.
    - device: Device to use for training (CPU/GPU).
    - phase: 'children' or 'parents', indicating the training phase.
    
    Returns:
    - avg_loss: Average loss over the dataset.
    """
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss

    for data in loader:
        data = data.to(device)  # Move data to the specified device
        optimizer.zero_grad()   # Reset gradients

        # Forward pass and compute loss
        out = model(data)
        loss = criterion(out[data["var_nodes"].mask],
                         data['var_nodes'].x_sub_opt[data["var_nodes"].mask])

        loss.backward()   # Backpropagation
        optimizer.step()  # Update model weights

        total_loss += loss.item() * data['var_nodes'].num_nodes  # Accumulate loss

    avg_loss = total_loss / len(loader.dataset)  # Calculate average loss

    return avg_loss



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




def test(model, loader, criterion, device):
    return